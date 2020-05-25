# Usage: python generate_experiments.py > experiments.sh && bash experiments.sh 
params = {
  "t-passes": [1, 3, 5],
  "i-passes": [1,3,5],
  "layers": [
    [100, 50],
    [15, 13, 11],
    [300, 150]
  ]
}

t_string = "bash run_circle.sh {} {} {}"

for t_passes in params['t-passes']:
    for i_passes in params['i-passes']:
        for layer_list in params['layers']:
            print(t_string.format(str(t_passes), str(i_passes), ' '.join(map(str, layer_list))))
