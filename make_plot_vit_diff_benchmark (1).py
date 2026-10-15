import matplotlib.pyplot as plt
import numpy as np
import math 

import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib as mpl
import argparse
from scipy.ndimage.filters import gaussian_filter1d
from collections import defaultdict

mpl.rcParams.update({
    'font.size'           : 9.0        ,
    'font.sans-serif'     : 'Arial'    ,
    'xtick.major.size'    : 2.0          ,
    'xtick.minor.size'    : 0        ,
    'xtick.major.width'   : 0.75       ,
    'xtick.minor.width'   : 0.75       ,
    'xtick.labelsize'     : 7.0        ,
    'xtick.direction'     : 'in'       ,
    'xtick.top'           : True       ,
    'ytick.major.size'    : 2          ,
    'ytick.minor.size'    : 1.5        ,
    'ytick.major.width'   : 0.75       ,
    'ytick.minor.width'   : 0.75       ,
    'ytick.labelsize'     : 7.0        ,
    'ytick.direction'     : 'in'       ,
    'xtick.major.pad'     : 1.5          ,
    'xtick.minor.pad'     : 1.5          ,
    'ytick.major.pad'     : 1.5          ,
    'ytick.minor.pad'     : 1.5          ,
    'ytick.right'         : True       ,
    'savefig.dpi'         : 600        ,
    'savefig.transparent' : True       ,
    'axes.linewidth'      : 0.75       ,
    'lines.linewidth'     : 1.0        ,
    'axes.prop_cycle'     : mpl.cycler('color', ['3030FF', '2ECC71', 'D04C3C', 'E3B448', '1B6535', 'B1624E', 'BDC3C7', 'CBCE91', '195190',
                                                'FF5733', '9B59B6', 'EF9DAF'])
})

##### input #####
num_max_shot = 8
benchmark = "TrueMICL"
IT_benchmark = "InternVL"
orderings = ["both_ascending", "both_descending", "random"] #"img_ascending", "img_descending", "text_ascending", "text_descending", 
max_shot_index = int(math.log(num_max_shot) + 3)
icl_examples = [0, 1, 2, 4, 8]
##################

def prepare_data(benchmark, IT_benchmark, num_max_shot, ordering): 

    results = {}
    if benchmark == "TrueMICL":
        results['Classification'] = {}
        results['Plaindrome'] = {}
        results['Shapes Count'] = {}
        # results['Sudoku'] = {}
        results['Operator Induction'] = {}
        results['CLEVR'] = {}
        results['Clock'] = {}

        if ordering == "random":
            results['Classification']['LLaVA-1.5'] = [32.9268, 28.0488, 26.8293]
            results['Classification']['QwenVL-2.5'] = [21.95, 25.6098, 25.6098]
            results['Classification']['InternVL-2.5'] = [45.122, 26.8293, 25.6098]

            results['Plaindrome']['LLaVA-1.5'] = [7, 11, 9, 13, 17]
            results['Plaindrome']['QwenVL-2.5'] = [75, 37, 17, 19, 10]
            results['Plaindrome']['InternVL-2.5'] = [27, 36, 47, 43, 45]

            results['Shapes Count']['LLaVA-1.5'] = [39, 16, 23, 30, 39]
            results['Shapes Count']['QwenVL-2.5'] = [7, 80, 50, 77, 80]
            results['Shapes Count']['InternVL-2.5'] = [5, 65, 73, 69, 61]

            # results['Sudoku']['LLaVA-1.5'] = [2, 0, 0, 0, 0]
            # results['Sudoku']['QwenVL-2.5'] = [97, 99, 65, 66, 60]
            # results['Sudoku']['InternVL-2.5'] = [76, 21, 82, 79, 72]

            results['Operator Induction']['LLaVA-1.5'] = [16, 5, 8, 11, 11]
            results['Operator Induction']['QwenVL-2.5'] = [30, 22, 10, 3, 4]
            results['Operator Induction']['InternVL-2.5'] = [12, 17, 12, 16, 6]

            results['CLEVR']['LLaVA-1.5'] = [50, 41.3043, 50, 54.3478, 47.8261]
            results['CLEVR']['QwenVL-2.5'] = [36.9565, 36.9565, 10.8696, 2.1739, 2.1739]
            results['CLEVR']['InternVL-2.5'] = [36.9565, 32.6087, 30.4348, 36.9565, 43.4783]

            results['Clock']['LLaVA-1.5'] = [6, 2, 6, 7, 7]
            results['Clock']['QwenVL-2.5'] = [36, 45, 46, 52, 47]
            results['Clock']['InternVL-2.5'] = [19, 17, 18, 13, 10]

        elif ordering == "img_ascending":
            results['Classification']['LLaVA-1.5'] = [32.9268, 17.8049, 21.7073]
            results['Classification']['QwenVL-2.5'] = [21.95, 25.6, 25.6]
            results['Classification']['InternVL-2.5'] = [45.122, 28.0488, 30.4878]

            results['Plaindrome']['LLaVA-1.5'] = [7, 11, 12, 13, 14]
            results['Plaindrome']['QwenVL-2.5'] = [75, 37, 16, 20, 17]
            results['Plaindrome']['InternVL-2.5'] = [27, 36, 50, 43, 45]

            results['Shapes Count']['LLaVA-1.5'] = [39, 16, 27, 38, 41]
            results['Shapes Count']['QwenVL-2.5'] = [7, 80, 58, 67, 82]
            results['Shapes Count']['InternVL-2.5'] = [5, 65, 69, 71, 64]

            # results['Sudoku']['LLaVA-1.5'] = [16, 5, 7, 9, 10]
            # results['Sudoku']['QwenVL-2.5'] = [97, 99, 74, 67, 74]
            # results['Sudoku']['InternVL-2.5'] = [76, 21, 76, 81, 79]

            results['Operator Induction']['LLaVA-1.5'] = [16, 5, 7, 9, 10]
            results['Operator Induction']['QwenVL-2.5'] = [30, 22, 8, 4, 4]
            results['Operator Induction']['InternVL-2.5'] = [12, 17, 17, 7, 7]

            results['CLEVR']['LLaVA-1.5'] = [50, 41.3043, 45.6522, 52.1739, 50]
            results['CLEVR']['QwenVL-2.5'] = [36.9565, 36.9565, 13.0435, 2.1739, 0]
            results['CLEVR']['InternVL-2.5'] = [36.9565, 32.6087, 30.4348, 47.8261, 43.4783]

            results['Clock']['LLaVA-1.5'] = [6, 2, 5, 3, 5]
            results['Clock']['QwenVL-2.5'] = [36, 45, 44, 48, 51]
            results['Clock']['InternVL-2.5'] = [19, 17, 25, 15, 9]

        elif ordering == "img_descending":
            results['Classification']['LLaVA-1.5'] = [32.9268, 0, 0]
            results['Classification']['QwenVL-2.5'] = [21.95, 26.8293, 34.1463]
            results['Classification']['InternVL-2.5'] = [45.122, 37.8049, 30.4878]

            results['Plaindrome']['LLaVA-1.5'] = [7, 11, 9, 15, 17]
            results['Plaindrome']['QwenVL-2.5'] = [75, 37, 21, 17, 16]
            results['Plaindrome']['InternVL-2.5'] = [27, 36, 46, 41, 43]

            results['Shapes Count']['LLaVA-1.5'] = [39, 16, 32, 36, 44]
            results['Shapes Count']['QwenVL-2.5'] = [7, 80, 1, 76, 83]
            results['Shapes Count']['InternVL-2.5'] = [5, 65, 70, 70, 63]

            # results['Sudoku']['LLaVA-1.5'] = []
            # results['Sudoku']['QwenVL-2.5'] = []
            # results['Sudoku']['InternVL-2.5'] = []

            results['Operator Induction']['LLaVA-1.5'] = [16, 5, 8, 9, 10]
            results['Operator Induction']['QwenVL-2.5'] = [97, 99, 69, 74, 77]
            results['Operator Induction']['InternVL-2.5'] = [12, 17, 11, 14, 4]

            results['CLEVR']['LLaVA-1.5'] = [50, 41.3043, 41.3043, 52.1739, 60.8696]
            results['CLEVR']['QwenVL-2.5'] = [36.9565, 36.9565, 10.8696, 4.3478, 2.1739]
            results['CLEVR']['InternVL-2.5'] = [36.9565, 32.6087, 34.7826, 39.1304, 41.3043]

            results['Clock']['LLaVA-1.5'] = [6, 2, 4, 8, 5]
            results['Clock']['QwenVL-2.5'] = [36, 45, 45, 44, 50]
            results['Clock']['InternVL-2.5'] = [19, 17, 25, 14, 13]

        elif ordering == "text_descending":
            results['Classification']['LLaVA-1.5'] = []
            results['Classification']['QwenVL-2.5'] = []
            results['Classification']['InternVL-2.5'] = [45.122, 34.1463, 32.9268]

            results['Plaindrome']['LLaVA-1.5'] = []
            results['Plaindrome']['QwenVL-2.5'] = []
            results['Plaindrome']['InternVL-2.5'] = [27, 36, 50, 48, 52]

            results['Shapes Count']['LLaVA-1.5'] = []
            results['Shapes Count']['QwenVL-2.5'] = []
            results['Shapes Count']['InternVL-2.5'] = [5, 65, 69, 68, 64]

            results['Operator Induction']['LLaVA-1.5'] = []
            results['Operator Induction']['QwenVL-2.5'] = []
            results['Operator Induction']['InternVL-2.5'] = [12, 17, 17, 10, 7]

            results['CLEVR']['LLaVA-1.5'] = []
            results['CLEVR']['QwenVL-2.5'] = []
            results['CLEVR']['InternVL-2.5'] = [36.9565, 32.6087, 30.4348, 41.3043, 43.4783]

            results['Clock']['LLaVA-1.5'] = []
            results['Clock']['QwenVL-2.5'] = []
            results['Clock']['InternVL-2.5'] = [19, 17, 16, 14, 8]

        elif ordering == "text_ascending":
            results['Classification']['LLaVA-1.5'] = [32.9268, 34.1463, 21.9512]
            results['Classification']['QwenVL-2.5'] = []
            results['Classification']['InternVL-2.5'] = [45.122, 30.4878, 34.1463]

            results['Plaindrome']['LLaVA-1.5'] = [7, 11, 10, 8, 5]
            results['Plaindrome']['QwenVL-2.5'] = []
            results['Plaindrome']['InternVL-2.5'] = [27, 36, 46, 48, 51]

            results['Shapes Count']['LLaVA-1.5'] = [39, 16, 30, 28, 26]
            results['Shapes Count']['QwenVL-2.5'] = []
            results['Shapes Count']['InternVL-2.5'] =[5, 65, 70, 61, 65]

            results['Operator Induction']['LLaVA-1.5'] = [16, 5, 5, 4, 3]
            results['Operator Induction']['QwenVL-2.5'] = []
            results['Operator Induction']['InternVL-2.5'] = [12, 17, 11, 12, 4]

            results['CLEVR']['LLaVA-1.5'] = [50, 41.3043, 39.1304, 32.6087, 34.7826]
            results['CLEVR']['QwenVL-2.5'] = []
            results['CLEVR']['InternVL-2.5'] = [36.9565, 32.6087, 34.7826, 34.7826, 41.3043]

            results['Clock']['LLaVA-1.5'] = [6, 2, 6, 7, 3]
            results['Clock']['QwenVL-2.5'] = []
            results['Clock']['InternVL-2.5'] = [19, 17, 14, 17, 9]

            
        elif ordering == "both_descending":
            results['Classification']['LLaVA-1.5'] = [32.9268, 30.4878, 31.7073]
            results['Classification']['QwenVL-2.5'] = [21.95, 22.9268, 30.4878]
            results['Classification']['InternVL-2.5'] = [45.122, 30.4878, 32.9268]

            results['Plaindrome']['LLaVA-1.5'] = [7, 11, 7, 8, 5]
            results['Plaindrome']['QwenVL-2.5'] = [75, 37, 15, 14, 12]
            results['Plaindrome']['InternVL-2.5'] = [27, 36, 50, 47, 45]

            results['Shapes Count']['LLaVA-1.5'] = [39, 16, 32, 30, 35]
            results['Shapes Count']['QwenVL-2.5'] = [7, 80, 55, 73, 77]
            results['Shapes Count']['InternVL-2.5'] = [5, 65, 69, 72, 65]
            
            # results['Sudoku']['LLaVA-1.5'] = []
            # results['Sudoku']['QwenVL-2.5'] = []
            # results['Sudoku']['InternVL-2.5'] = []

            results['Operator Induction']['LLaVA-1.5'] = [16, 5, 4, 4, 2]
            results['Operator Induction']['QwenVL-2.5'] = [30, 22, 7, 4, 2]
            results['Operator Induction']['InternVL-2.5'] = [12, 17, 17, 10, 7]

            results['CLEVR']['LLaVA-1.5'] = [50, 41.3043, 47.8261, 56.5217, 45.6522]
            results['CLEVR']['QwenVL-2.5'] = [36.9565, 36.9565, 13.0435, 2.1739, 13.0435]
            results['CLEVR']['InternVL-2.5'] = [36.9565, 32.6087, 30.4348, 36.9565, 43.4783]

            results['Clock']['LLaVA-1.5'] = [6, 2, 6, 6, 3]
            results['Clock']['QwenVL-2.5'] = [36, 45, 40, 42, 45]
            results['Clock']['InternVL-2.5'] = [19, 17, 17, 9, 8]

        elif ordering == "both_ascending":
            results['Classification']['LLaVA-1.5'] = [32.9268, 30.4878, 19.5122]
            results['Classification']['QwenVL-2.5'] = [21.95, 26.8293, 32.9268]
            results['Classification']['InternVL-2.5'] = [45.122, 32.9268, 34.1463]

            results['Plaindrome']['LLaVA-1.5'] = [7, 11, 10, 8, 4]
            results['Plaindrome']['QwenVL-2.5'] = [75, 37, 15, 20, 17]
            results['Plaindrome']['InternVL-2.5'] = [27, 36, 46, 41, 43]

            results['Shapes Count']['LLaVA-1.5'] = [39, 16, 30, 28, 31]
            results['Shapes Count']['QwenVL-2.5'] = [7, 80, 62, 78, 80]
            results['Shapes Count']['InternVL-2.5'] = [5, 65, 70, 64, 60]

            results['Operator Induction']['LLaVA-1.5'] = [16, 5, 5, 6, 4]
            results['Operator Induction']['QwenVL-2.5'] = [30, 22, 10, 4, 4]
            results['Operator Induction']['InternVL-2.5'] = [12, 17, 11, 14, 4]

            results['CLEVR']['LLaVA-1.5'] = [50, 41.3043, 39.1304, 54.3478, 56.2922]
            results['CLEVR']['QwenVL-2.5'] = [36.9565, 36.9565, 34.7826, 2.1739, 0]
            results['CLEVR']['InternVL-2.5'] = [36.9565, 32.6087, 34.7826, 39.1304, 41.3043]

            results['Clock']['LLaVA-1.5'] = [6, 2, 6, 9, 2]
            results['Clock']['QwenVL-2.5'] = [36, 45, 45, 47, 41]
            results['Clock']['InternVL-2.5'] = [19, 17, 13, 12, 4]


    elif benchmark == "VL-ICL":
        max_shot_index = int(math.log(num_max_shot) + 2)
        icl_examples = [1, 2, 4, 8]

        results['CLEVR'] = {}
        results['TextOCR'] = {}
        results['Operator Induction'] = {}
        results['Operator Induction Interleaved'] = {}

        results['CLEVR']['LLaVA-1.5'] = [31.5, 26, 19, 20]
        results['CLEVR']['QwenVL-2.5'] = [27, 18.5, 14, 6]
        results['CLEVR']['InternVL-2.5'] = [25, 26, 19, 20]

        results['TextOCR']['LLaVA-1.5'] = [42.5, 45.5, 46, 47.5]
        results['TextOCR']['QwenVL-2.5'] = [44.5, 46.5, 47.5, 47]
        results['TextOCR']['InternVL-2.5'] = [24.5, 45.5, 46, 47.5]

        results['Operator Induction']['LLaVA-1.5'] = [41.6667, 35, 40, 25]
        results['Operator Induction']['QwenVL-2.5'] = [10, 11.6667, 6.6667, 5]
        results['Operator Induction']['InternVL-2.5'] = [5, 35, 40, 25]

        results['Operator Induction Interleaved']['LLaVA-1.5'] = [10, 11.6667, 8.3333, 3.3333]
        results['Operator Induction Interleaved']['QwenVL-2.5'] = [16.666, 8.3333, 1.6667, 0]
        results['Operator Induction Interleaved']['InternVL-2.5'] = [10, 11.6667, 8.3333, 3.3333]

    return results

def draw_plots(results, task_y_max, task_y_min, benchmark, IT_benchmark, num_max_shot, ordering):
    # Plot 그리기
    for task in results.keys(): # key: subtask
        plt.figure(figsize=(8, 6))
        for model in results[task].keys():
            if task == 'Classification':
                plt.plot([0, 4, 8], np.array(results[task][model]), marker='o', linewidth=2.5, label=model)
            else:
                plt.plot(np.array(icl_examples)[:max_shot_index], np.array(results[task][model])[:max_shot_index], marker='o', linewidth=2.5, label=model)

        plt.xlabel('# of In-Context Examples', fontsize=26)
        plt.ylabel('ROUGE-L', fontsize=26)
        
        plt.tick_params(axis='y', labelsize=25)
        plt.xticks([4, 8], fontsize=25)  # x축 눈금 수동 설정
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        plt.ylim(task_y_min[task]-5, task_y_max[task]+5)
        plt.title(f"{task}", fontsize=30)
        plt.legend(fontsize=25)
        plt.grid(True)
        plt.tight_layout()

        # 그래프 보여주기
        plt.savefig(f"{benchmark}_{task}_{ordering}_scenario.png")

tmp_result = []
task_y_min = defaultdict(lambda: 1000)
task_y_max = defaultdict(lambda: -1000)

for ordering in orderings:
    tmp_result.append(prepare_data(benchmark, IT_benchmark, num_max_shot, ordering))

for result in tmp_result:
    for task in result.keys():
        all_vals = [v for vals in result[task].values() for v in vals]
        task_y_min[task] = min(task_y_min[task], min(all_vals))
        task_y_max[task] = max(task_y_min[task], max(all_vals))

for ordering, result in zip(orderings, tmp_result):
    draw_plots(result, task_y_max, task_y_min, benchmark, IT_benchmark, num_max_shot, ordering)