import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr
 
try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.sampling import Sampling
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.optimize import minimize
except ImportError:
    raise ImportError("pip")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_FOLDER = 'data'
INPUT_FILE = os.path.join(DATA_FOLDER, 'laptop_processed.csv')
OUTPUT_FOLDER = 'results'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def repair_vector_logic(sol, n_items, max_unique_models, perfs, num_products):
    indices = np.where(sol > 0)[0]
    if len(indices) > max_unique_models:
        top_indices = sorted(indices, key=lambda i: sol[i], reverse=True)[:max_unique_models]
        new_sol = np.zeros_like(sol)
        for i in top_indices:
            new_sol[i] = sol[i]
        sol = new_sol

    current_sum = np.sum(sol)
    
    while current_sum > n_items:
        indices = np.where(sol > 0)[0]
        if len(indices) > 0:
            idx = random.choice(indices)
            sol[idx] -= 1
            current_sum -= 1
        else: break
        
    while current_sum < n_items:
        indices = np.where(sol > 0)[0]
        if len(indices) > 0:
            idx = random.choice(indices)
        else:
            top_models = np.argsort(perfs)[-max_unique_models:]
            idx = random.choice(top_models)
        
        sol[idx] += 1
        current_sum += 1
        
    return sol

class LaptopProblem(ElementwiseProblem):
    def __init__(self, df, n_items, budget, max_unique_models):
        self.df = df
        self.n_items = n_items
        self.budget = budget
        self.max_unique_models = max_unique_models
        
        self.perfs = df['Performance_Score'].values
        self.prices = df['Price_VND'].values
        self.energies = df['TDP'].values
        
        super().__init__(n_var=len(df), n_obj=3, n_ieq_constr=1)

    def _evaluate(self, x, out, *args, **kwargs):
        counts = x.astype(int)
        
        f1 = -np.sum(counts * self.perfs)  # Max Perf -> Min -Perf
        f2 = np.sum(counts * self.prices)  # Min Price
        f3 = np.sum(counts * self.energies) # Min Energy
        g1 = f2 - self.budget 

        out["F"] = [f1, f2, f3]
        out["G"] = [g1]

class CustomSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        for i in range(n_samples):
            sol = np.zeros(problem.n_var, dtype=int)
            if problem.max_unique_models < problem.n_var:
                selected_models = np.random.choice(problem.n_var, problem.max_unique_models, replace=False)
                for _ in range(problem.n_items): 
                    sol[random.choice(selected_models)] += 1
            else:
                for _ in range(problem.n_items): 
                    sol[random.randint(0, problem.n_var-1)] += 1
            X[i, :] = sol
        return X

class CustomCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)
    
    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        Y = np.zeros_like(X)
        for k in range(n_matings):
            parent1, parent2 = X[0, k], X[1, k]
            cut = random.randint(1, n_var - 2)
            child1 = np.concatenate((parent1[:cut], parent2[cut:]))
            child2 = np.concatenate((parent2[:cut], parent1[cut:]))
            
            Y[0, k] = repair_vector_logic(child1, problem.n_items, problem.max_unique_models, problem.perfs, n_var)
            Y[1, k] = repair_vector_logic(child2, problem.n_items, problem.max_unique_models, problem.perfs, n_var)
        return Y

class CustomMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            if random.random() < 0.1:
                idx1, idx2 = random.sample(range(problem.n_var), 2)
                X[i][idx1], X[i][idx2] = X[i][idx2], X[i][idx1]
                X[i] = repair_vector_logic(X[i], problem.n_items, problem.max_unique_models, problem.perfs, problem.n_var)
        return X

class NSGA2_Optimizer_Wrapper:
    def __init__(self, df, n_items, budget, max_unique_models):
        self.df = df
        self.n_items = n_items
        self.budget = budget
        self.max_unique_models = max_unique_models

    def run(self, pop_size=200, generations=100):
        print(f"(N={self.n_items}, Budget={self.budget:,.0f})...")
        
        problem = LaptopProblem(self.df, self.n_items, self.budget, self.max_unique_models)
        
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=CustomSampling(),
            crossover=CustomCrossover(),
            mutation=CustomMutation(),
            eliminate_duplicates=True
        )
        
        res = minimize(problem,
                       algorithm,
                       ('n_gen', generations),
                       seed=SEED,
                       verbose=True)
        
        if res.G is not None:
            feasible_indices = np.where(res.G[:, 0] <= 0)[0]
            final_X = res.X[feasible_indices]
            final_F = res.F[feasible_indices]
        else:
            final_X = res.X
            final_F = res.F

        class PymooIndWrapper:
            def __init__(self, x, f):
                self.solution = x
                self.fitness = f
        
        front = [PymooIndWrapper(x, f) for x, f in zip(final_X, final_F)]
        
        return front

class ResultAnalyzer:
    def __init__(self, optimizer, pareto_front, label):
        self.opt = optimizer
        self.front = pareto_front
        self.label = label
        
        data = []
        for ind in self.front:
            indices = np.where(ind.solution > 0)[0]
            desc_list = []
            for i in indices:
                qty = ind.solution[i]
                row = self.opt.df.iloc[i]
                detail_str = (f"{qty}x {row['Model']}<br>"
                              f"&nbsp;&nbsp;[Gia: {row['Price_VND']:,.0f} | RAM: {row['Ram']}GB | "
                              f"SSD: {row['SSD']}GB | G3D: {row['G3Dmark']} | TDP: {row['TDP']}W]")
                desc_list.append(detail_str)
            
            data.append({
                'Total_Perf': -ind.fitness[0],
                'Total_Price': ind.fitness[1],
                'Total_Energy': ind.fitness[2],
                'Solution_Vector': ind.solution,
                'Details_Tooltip': "<br>".join(desc_list),
                'Num_Models': len(indices) 
            })
        self.df_res = pd.DataFrame(data)

    def identify_scenarios(self):
        if self.df_res.empty:
            return None
        
        s_budget = self.df_res.loc[self.df_res['Total_Price'].idxmin()]
        s_perf = self.df_res.loc[self.df_res['Total_Perf'].idxmax()]
        
        p_min = self.df_res['Total_Price'].min()
        p_max = self.df_res['Total_Price'].max()
        perf_min = self.df_res['Total_Perf'].min()
        perf_max = self.df_res['Total_Perf'].max()
        
        norm_price = (self.df_res['Total_Price'] - p_min) / (p_max - p_min + 1e-6)
        norm_perf = (self.df_res['Total_Perf'] - perf_min) / (perf_max - perf_min + 1e-6)
        
        dist = np.sqrt(norm_price**2 + (1 - norm_perf)**2)
        s_balanced = self.df_res.loc[dist.idxmin()]
        
        return {
            'Budget': s_budget, 
            'Performance': s_perf, 
            'Balanced': s_balanced
        }

    def export_csv(self):
        filename = os.path.join(OUTPUT_FOLDER, f'pareto_front_{self.label}.csv')
        
        df_export = self.df_res.drop(columns=['Solution_Vector', 'Details_Tooltip'])
        df_export.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f" Đã xuất CSV: {filename}")

    def plot_static_chart_3d(self, scenarios):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
   
        ax.scatter(
            self.df_res['Total_Price'], 
            self.df_res['Total_Perf'], 
            self.df_res['Total_Energy'],
            c='lightgray', 
            alpha=0.6, 
            s=60, 
            marker='o', 
            label='Pareto Front', 
            edgecolors='darkgray'
        )
        
        colors = {'Budget': 'green', 'Performance': 'red', 'Balanced': 'blue'}
        markers = {'Budget': 'v', 'Performance': '^', 'Balanced': '*'}
        
        for name, row in scenarios.items():
            if row['Total_Price'] >= 1e9:
                price_str = f"{row['Total_Price']/1e9:.1f}B"
            else:
                price_str = f"{row['Total_Price']/1e6:.0f}M"
            
            ax.scatter(
                row['Total_Price'], 
                row['Total_Perf'], 
                row['Total_Energy'],
                c=colors[name], 
                s=300, 
                marker=markers[name], 
                label=f'{name} ({price_str})', 
                edgecolors='black', 
                linewidths=1.5
            )
            
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x*1e-9:.1f}B' if x >= 1e9 else f'{x*1e-6:.0f}M'))
        
        ax.set_xlabel('Tổng Giá')
        ax.set_ylabel('Tổng Hiệu Năng')
        ax.set_zlabel('Tổng Năng Lượng')
        ax.set_title(f'Không gian 3D ({self.label})')
        
        ax.view_init(elev=25, azim=135)
        ax.legend()
        
        plt.savefig(os.path.join(OUTPUT_FOLDER, f'3d_chart_{self.label}.png'), dpi=150)
        plt.close()
        

    def plot_dynamic_chart_3d(self, scenarios):
        self.df_res['Scenario_Type'] = 'Pareto Solution'
  
        for name, row in scenarios.items():
            self.df_res.loc[row.name, 'Scenario_Type'] = name
            
        fig = px.scatter_3d(
            self.df_res, 
            x='Total_Price', 
            y='Total_Perf', 
            z='Total_Energy', 
            color='Scenario_Type',
            hover_data={
                'Details_Tooltip': True, 
                'Total_Price': ':,.0f', 
                'Total_Perf': ':.2f', 
                'Total_Energy': ':.2f', 
                'Scenario_Type': False, 
                'Solution_Vector': False
            },
            title=f"Biểu đồ 3D ({self.label})", 
            color_discrete_map={
                'Pareto Solution': 'gray', 
                'Budget': 'green', 
                'Performance': 'red', 
                'Balanced': 'blue'
            }
        )
        
        fig.update_scenes(xaxis_autorange="reversed", zaxis_autorange="reversed")
        fig.update_traces(hovertemplate="<b>%{data.name}</b><br>Giá: %{x:,.0f}<br>Perf: %{y:.2f}<br>Energy: %{z:.2f}<br><b>%{customdata[0]}</b><extra></extra>")
        
        fig.write_html(os.path.join(OUTPUT_FOLDER, f'3d_interactive_{self.label}.html'))

    def plot_tradeoff_analysis(self):
        if len(self.df_res) <= 1:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pt1: Giá vs Hiệu năng
        ax1 = axes[0]
        ax1.scatter(self.df_res['Total_Price'], self.df_res['Total_Perf'], alpha=0.6, c='steelblue')
        
        z = np.polyfit(self.df_res['Total_Price'], self.df_res['Total_Perf'], 1)
        p = np.poly1d(z)
        corr, _ = pearsonr(self.df_res['Total_Price'], self.df_res['Total_Perf'])
        
        ax1.plot(self.df_res['Total_Price'], p(self.df_res['Total_Price']), "r--", label=f'R={corr:.3f}')
        ax1.set_xlabel('Tổng Giá')
        ax1.set_ylabel('Tổng Hiệu Năng')
        ax1.legend()
        ax1.grid(True)
        
        # Pt2: Hiệu năng vs Năng lượng
        ax2 = axes[1]
        ax2.scatter(self.df_res['Total_Perf'], self.df_res['Total_Energy'], alpha=0.6, c='coral')
        
        z2 = np.polyfit(self.df_res['Total_Perf'], self.df_res['Total_Energy'], 1)
        p2 = np.poly1d(z2)
        corr2, _ = pearsonr(self.df_res['Total_Perf'], self.df_res['Total_Energy'])
        
        ax2.plot(self.df_res['Total_Perf'], p2(self.df_res['Total_Perf']), "g--", label=f'R={corr2:.3f}')
        ax2.set_xlabel('Tổng Hiệu Năng')
        ax2.set_ylabel('Tổng Năng Lượng')
        ax2.legend()
        ax2.grid(True)
        
        plt.savefig(os.path.join(OUTPUT_FOLDER, f'tradeoff_analysis_{self.label}.png'), dpi=150)
        plt.close()
        
        print(f"\n TƯƠNG QUAN ({self.label}):\n   - Giá - Hiệu năng: r = {corr:.3f}\n   - Hiệu năng - Năng lượng: r = {corr2:.3f}")

def main():
    if not os.path.exists(INPUT_FILE): print(f"Can not find {INPUT_FILE}"); return
    df = pd.read_csv(INPUT_FILE)
    if len(df) == 0: return
    
    configs = [
        {'n': 1,  'budget': 50_000_000,    'label': 'N1_Personal',   'max_models': 1, 'pop': 50, 'gen': 30},
        {'n': 20, 'budget': 1_000_000_000, 'label': 'N20_SmallTeam', 'max_models': 3, 'pop': 200, 'gen': 100}
    ]
    
    for cfg in configs:
        print(f"\n {cfg['label']}: N={cfg['n']}, Budget={cfg['budget']:,.0f}")
        opt = NSGA2_Optimizer_Wrapper(df, n_items=cfg['n'], budget=cfg['budget'], max_unique_models=cfg['max_models'])
        front = opt.run(pop_size=cfg['pop'], generations=cfg['gen'])
        
        analyzer = ResultAnalyzer(opt, front, label=cfg['label'])
        scenarios = analyzer.identify_scenarios()
        
        if scenarios:
            analyzer.export_csv()
            analyzer.plot_static_chart_3d(scenarios)
            analyzer.plot_dynamic_chart_3d(scenarios)
            analyzer.plot_tradeoff_analysis()
            
            print(f"\n Initial:")
            for s_name, s_row in scenarios.items():
                print(f"   * {s_name}: {s_row['Total_Price']:,.0f} VNĐ | Perf={s_row['Total_Perf']:.2f}")
        else: print(" None")

if __name__ == "__main__":
    main()