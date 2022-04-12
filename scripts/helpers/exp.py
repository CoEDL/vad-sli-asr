import os

def output_path_maker(exp_name, results_dir):
    def path_maker(filename):
        return os.path.join(results_dir, exp_name + "_" + filename)

    return path_maker
