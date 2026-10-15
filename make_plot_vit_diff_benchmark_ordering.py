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
benchmark = "Standford"
architectures = ['InternVL', 'LLaVA-1.5', 'Qwen-VL']
##################
max_shot_index = int(math.log(num_max_shot) + 3)
icl_examples = [0, 4, 8 , 12]
# orderings = ["both_ascending", "both_descending", "random"] #"img_ascending", "img_descending", "text_ascending", "text_descending", 

def prepare_data(benchmark, num_max_shot, architecture): 
    results = {}
    if benchmark == "TrueMICL":
        results['Classification'] = {}
        results['Plaindrome'] = {}
        results['Shapes Count'] = {}
        # results['Sudoku'] = {}
        results['Operator Induction'] = {}
        results['CLEVR'] = {}
        results['Clock'] = {}

        if architecture == "LLaVA-1.5":
            results['Classification']['random'] = [32.9268, 28.0488, 26.8293]
            # results['Classification']['descending'] = [32.9268, 0, 0]
            # results['Classification']['ascending'] = [32.9268, 17.8049, 21.7073]
            results['Classification']['descending'] = [32.9268, 30.4878, 31.7073]
            results['Classification']['ascending'] = [32.9268, 30.4878, 19.5122]

            results['Plaindrome']['random'] = [7, 11, 9, 13, 17]
            # results['Plaindrome']['descending'] = [7, 11, 9, 15, 17]
            # results['Plaindrome']['ascending'] = [7, 11, 12, 13, 14]
            results['Plaindrome']['descending'] = [7, 11, 7, 8, 5]
            results['Plaindrome']['ascending'] = [7, 11, 10, 8, 4]

            results['Shapes Count']['random'] = [39, 16, 23, 30, 39]
            # results['Shapes Count']['descending'] = [39, 16, 32, 36, 44]
            # results['Shapes Count']['ascending'] = [39, 16, 27, 38, 41]
            results['Shapes Count']['descending'] = [39, 16, 32, 30, 35]
            results['Shapes Count']['ascending'] = [39, 16, 30, 28, 31]

            results['Operator Induction']['random'] = [16, 5, 8, 11, 11]
            # results['Operator Induction']['descending'] = [16, 5, 8, 9, 10]
            # results['Operator Induction']['ascending'] = [16, 5, 7, 9, 10]
            results['Operator Induction']['descending'] = [16, 5, 4, 4, 2]
            results['Operator Induction']['ascending'] = [16, 5, 5, 6, 4]

            results['CLEVR']['random'] = [50, 41.3043, 50, 54.3478, 47.8261]
            # results['CLEVR']['descending'] = [50, 41.3043, 41.3043, 52.1739, 60.8696]
            # results['CLEVR']['ascending'] = [50, 41.3043, 45.6522, 52.1739, 50]
            results['CLEVR']['descending'] = [50, 41.3043, 47.8261, 56.5217, 45.6522]
            results['CLEVR']['ascending'] = [50, 41.3043, 39.1304, 54.3478, 56.2922]

            results['Clock']['random'] = [6, 2, 6, 7, 7]
            # results['Clock']['descending'] = [6, 2, 4, 8, 5]
            # results['Clock']['ascending'] = [6, 2, 5, 3, 5]
            results['Clock']['descending'] = [6, 2, 6, 6, 3]
            results['Clock']['ascending'] = [6, 2, 6, 9, 2]

        elif architecture == "Qwen-VL":

            results['Classification']['random'] = [21.95, 25.6098, 25.6098]
            # results['Classification']['descending'] = [21.95, 26.8293, 34.1463]
            # results['Classification']['ascending'] = [21.95, 25.6, 25.6]
            results['Classification']['descending'] = [21.95, 22.9268, 30.4878]
            results['Classification']['ascending'] = [45.122, 30.4878, 34.1463]

            results['Plaindrome']['random'] = [75, 37, 17, 19, 10]
            # results['Plaindrome']['descending'] = [75, 37, 21, 17, 16]
            # results['Plaindrome']['ascending'] = [75, 37, 16, 20, 17]
            results['Plaindrome']['descending'] = [75, 37, 15, 14, 12]
            results['Plaindrome']['ascending'] = [27, 36, 46, 48, 51]

            results['Shapes Count']['random'] = [7, 80, 50, 77, 80]
            # results['Shapes Count']['descending'] = [7, 80, 1, 76, 83]
            # results['Shapes Count']['ascending'] = [7, 80, 58, 67, 82]
            results['Shapes Count']['descending'] = [7, 80, 55, 73, 77]
            results['Shapes Count']['ascending'] = [5, 65, 70, 61, 65]

            # results['Sudoku']['random'] = [97, 99, 65, 66, 60]
            # results['Sudoku']['descending'] = [97, 99, 69, 74, 77]
            # results['Sudoku']['ascending'] = [97, 99, 74, 67, 74]

            results['Operator Induction']['random'] = [30, 22, 10, 3, 4]
            # results['Operator Induction']['descending'] = [30, 22, 13, 6, 6]
            # results['Operator Induction']['ascending'] = [30, 22, 8, 4, 4]
            results['Operator Induction']['descending'] = [30, 22, 7, 4, 2]
            results['Operator Induction']['ascending'] = [12, 17, 11, 12, 4]

            results['CLEVR']['random'] = [36.9565, 36.9565, 10.8696, 2.1739, 2.1739]
            # results['CLEVR']['descending'] = [36.9565, 36.9565, 10.8696, 4.3478, 2.1739]
            # results['CLEVR']['ascending'] = [36.9565, 36.9565, 13.0435, 2.1739, 0]
            results['CLEVR']['descending'] = [36.9565, 36.9565, 13.0435, 2.1739, 13.0435]
            results['CLEVR']['ascending'] = [36.9565, 32.6087, 34.7826, 34.7826, 41.3043]

            results['Clock']['random'] = [36, 45, 46, 52, 47]
            # results['Clock']['descending'] = [36, 45, 45, 44, 50]
            # results['Clock']['ascending'] = [36, 45, 44, 48, 51]
            results['Clock']['descending'] = [36, 45, 40, 42, 45]
            results['Clock']['ascending'] = [19, 17, 14, 17, 9]

        elif architecture == "InternVL":
            results['Classification']['random'] = [45.122, 26.8293, 25.6098]
            # results['Classification']['descending'] = [45.122, 37.8049, 30.4878]
            # results['Classification']['ascending'] = [45.122, 28.0488, 30.4878]
            results['Classification']['descending'] = [45.122, 30.4878, 32.9268]
            results['Classification']['ascending'] = [45.122, 32.9268, 34.1463]

            results['Plaindrome']['random'] = [27, 36, 47, 43, 45]
            # results['Plaindrome']['descending'] = [27, 36, 46, 41, 43]
            # results['Plaindrome']['ascending'] = [27, 36, 50, 43, 45]
            results['Plaindrome']['descending'] = [27, 36, 50, 47, 45]
            results['Plaindrome']['ascending'] = [27, 36, 46, 41, 43]

            results['Shapes Count']['random'] = [5, 65, 73, 69, 61]
            # results['Shapes Count']['descending'] = [5, 65, 70, 70, 63]
            # results['Shapes Count']['ascending'] = [5, 65, 69, 71, 64]
            results['Shapes Count']['descending'] = [5, 65, 69, 72, 65]
            results['Shapes Count']['ascending'] = [5, 65, 70, 64, 60]

            # results['Sudoku']['random'] = [76, 21, 82, 79, 72]
            # results['Sudoku']['descending'] = [76, 21, 78, 79, 74]
            # results['Sudoku']['ascending'] = [76, 21, 76, 81, 79]
            # results['Sudoku']['descending'] = []
            # results['Sudoku']['ascending'] = []

            results['Operator Induction']['random'] = [12, 17, 12, 16, 6]
            # results['Operator Induction']['descending'] = [12, 17, 11, 14, 4]
            # results['Operator Induction']['ascending'] = [12, 17, 17, 7, 7]
            results['Operator Induction']['descending'] = [12, 17, 17, 10, 7]
            results['Operator Induction']['ascending'] = [12, 17, 11, 14, 4]

            results['CLEVR']['random'] = [36.9565, 32.6087, 30.4348, 36.9565, 43.4783]
            # results['CLEVR']['descending'] = [36.9565, 32.6087, 34.7826, 39.1304, 41.3043]
            # results['CLEVR']['ascending'] = [36.9565, 32.6087, 30.4348, 47.8261, 43.4783]
            results['CLEVR']['descending'] = [36.9565, 32.6087, 30.4348, 36.9565, 41.3043]
            results['CLEVR']['ascending'] = [36.9565, 32.6087, 34.7826, 39.1304, 43.4783]

            results['Clock']['random'] = [19, 17, 18, 13, 10]
            # results['Clock']['descending'] = [19, 17, 25, 14, 13]
            # results['Clock']['ascending'] = [19, 17, 25, 15, 9]
            results['Clock']['descending'] = [19, 17, 17, 9, 8]
            results['Clock']['ascending'] = [19, 17, 13, 12, 4]

    elif benchmark == "VL-ICL":
        results['CLEVR'] = {}
        results['TextOCR'] = {}
        results['Operator Induction'] = {}
        results['Operator Induction Interleaved'] = {}

        if architecture == "LLaVA-1.5":
            results['CLEVR']['random'] = [31.5, 26, 19, 20]
            results['CLEVR']['descending'] = [31.5, 27.5, 24, 4]
            results['CLEVR']['ascending'] = [31.5, 27, 21, 19]

            results['TextOCR']['random'] = [42.5, 45.5, 46, 47.5]
            results['TextOCR']['descending'] = [42.5, 5.5, 10.5, 0]
            results['TextOCR']['ascending'] = [42.5, 38, 41, 30.5]

            results['Operator Induction']['random'] = [41.6667, 35, 40, 25]
            results['Operator Induction']['descending'] = [41.6667, 8.3333, 6.6667, 5]
            results['Operator Induction']['ascending'] = [41.6667, 21.6667, 26.6667, 21.6667]

            results['Operator Induction Interleaved']['random'] = [10, 11.6667, 8.3333, 3.3333]
            results['Operator Induction Interleaved']['descending'] = [10, 3.3333, 6.6667, 10]
            results['Operator Induction Interleaved']['ascending'] = [10, 3.3333, 5, 3.3333]

        elif architecture == "Qwen-VL":

            results['CLEVR']['random'] = [27, 18.5, 14, 6]
            results['CLEVR']['descending'] = [27, 27, 1, 7.5]
            results['CLEVR']['ascending'] = [27, 27.5, 4, 19]

            results['TextOCR']['random'] = [44.5, 46.5, 47.5, 47]
            results['TextOCR']['descending'] = [44.5, 49.5, 48, 47]
            results['TextOCR']['ascending'] = [44.5, 46.5, 42.5, 47]

            results['Operator Induction']['random'] = [10, 11.6667, 6.6667, 5]
            results['Operator Induction']['descending'] = [10, 8.3333, 0, 6.6667]
            results['Operator Induction']['ascending'] = [10, 8.3333, 6.6667, 5]

            results['Operator Induction Interleaved']['random'] = [16.666, 8.3333, 1.6667, 0]
            results['Operator Induction Interleaved']['descending'] = [16.666, 13.3333, 0, 0]
            results['Operator Induction Interleaved']['ascending'] = [16.666, 13.3333, 6.6667, 10]


        elif architecture == "InternVL":

            results['CLEVR']['random'] = [25, 26, 19, 20]
            results['CLEVR']['descending'] = [25, 27, 21, 19]
            results['CLEVR']['ascending'] = [25, 7.5, 19.5, 21]

            results['TextOCR']['random'] = [24.5, 45.5, 46, 47.5]
            results['TextOCR']['descending'] = [24.5, 42.5, 41, 30.5]
            results['TextOCR']['ascending'] = [24.5, 28, 32.5, 18.5]

            results['Operator Induction']['random'] = [5, 35, 40, 25]
            results['Operator Induction']['descending'] = [5, 21.6667, 26.6667, 21.6667]
            results['Operator Induction']['ascending'] = [5, 6.6667, 11.6667, 21.6667]

            results['Operator Induction Interleaved']['random'] = [10, 11.6667, 8.3333, 3.3333]
            results['Operator Induction Interleaved']['descending'] = [10, 3.3333, 6.6667, 3.3333]
            results['Operator Induction Interleaved']['ascending'] = [10, 11.6667, 8.33, 10]
            
    elif benchmark == "Standford":
        results['Standford'] = {}

        if architecture == "Qwen-VL":
            results['Standford']['random'] = [0,30.6,29, 21.4]
            results['Standford']['ans front'] = [0,22.8,18.2,17.6]
            results['Standford']['ans back'] = [0,36.8,37.4, 26.5]

        elif architecture == "LLaVA-1.5":

            results['Standford']['random'] = [2.5,12.5,7.6, 6.6]
            results['Standford']['ans front'] = [2.5,14.4,7.2, 6.8]
            results['Standford']['ans back'] = [2.5,16.7,19.6, 9.8]


        elif architecture == "InternVL":

            results['Standford']['random'] = [10.2, 44,47.2, 31.8]
            results['Standford']['ans front'] = [10.2,48.6,51.6, 33.5]
            results['Standford']['ans back'] = [10.2,52,45.8, 38.4]

    return results

def draw_plots(results, task_y_max, task_y_min, benchmark, num_max_shot, architecture):
    # Plot 그리기
    for task in results.keys(): # key: subtask
        plt.figure(figsize=(8, 6))
        for model in results[task].keys():
            if task == 'Classification':
                plt.plot([0, 4, 8, 12], np.array(results[task][model]), marker='o', linewidth=2.5, label=model)
            else:
                plt.plot(np.array(icl_examples)[:max_shot_index], np.array(results[task][model])[:max_shot_index], marker='o', linewidth=2.5, label=model)

        plt.xlabel('# of In-Context Examples', fontsize=26)
        plt.ylabel('Accuacy', fontsize=26)
        
        plt.tick_params(axis='y', labelsize=25)
        plt.xticks([4, 8, 12], fontsize=25)  # x축 눈금 수동 설정
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        # plt.ylim(task_y_min[task]-5, task_y_max[task]+5)
        plt.title(f"{architecture}", fontsize=30)
        plt.legend(fontsize=25)
        plt.grid(True)
        plt.tight_layout()

        # 그래프 보여주기
        plt.savefig(f"{benchmark}_{task}_{architecture}_order_comparison.png")


# draw_plots(benchmark, architecture, num_max_shot)


tmp_result = []
task_y_min = defaultdict(lambda: 1000)
task_y_max = defaultdict(lambda: -1000)

for architecture in architectures:
    tmp_result.append(prepare_data(benchmark, num_max_shot, architecture))

for result in tmp_result:
    for task in result.keys():
        all_vals = [v for vals in result[task].values() for v in vals]
        task_y_min[task] = min(task_y_min[task], min(all_vals))
        task_y_max[task] = max(task_y_min[task], max(all_vals))

for architecture, result in zip(architectures, tmp_result):
    draw_plots(result, task_y_max, task_y_min, benchmark, num_max_shot, architecture)