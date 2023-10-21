import time
from function import Function, Domain
from plotting import plot_experiment_results, plot_single_result
from solver import Solver, Result
import autograd.numpy as np


def single_experiment(
    f: Function,
    domain: Domain,
    solver: Solver,
    log: bool = False,
    log_interval_time=1,
    starting_point=None,
):
    res = solver.solve(
        f,
        starting_point if starting_point else domain.generate_random_vector(),
        domain,
        log=log,
        log_interval_time=log_interval_time,
    )
    plot_single_result(res, f, domain)
    print(res)
    print("\n")


def experiment_step_sizes(
    f: Function,
    domain: Domain,
    step_sizes: list,
    conditions_list: list,
    n_starting_points: int = 10,
):
    print("Generating starting points...")
    starting_points = [
        domain.generate_random_vector() for _ in range(n_starting_points)
    ]
    print("Starting points: ", starting_points)

    min_f_list = []
    best = Result(
        stop_condition=None,
        x0=None,
        x=None,
        f_value=np.inf,
        gradient_value=None,
        n_iter=0,
        time_running=0,
        history=[],
    )

    worst = Result(
        stop_condition=None,
        x0=None,
        x=None,
        f_value=-np.inf,
        gradient_value=None,
        n_iter=0,
        time_running=0,
        history=[],
    )

    total = len(step_sizes) * len(conditions_list) * n_starting_points
    i = 1
    time_start = time.time()
    for step_size in step_sizes:
        for conditions in conditions_list:
            min_f_list_local = []
            for starting_point in starting_points:
                current_time = time.time()
                eta = (current_time - time_start) / i * (total - i)
                eta_str = time.strftime("%M:%S", time.gmtime(eta))
                print("\n")
                print(
                    f"({i}/{total}) [{eta_str}] step_size={step_size} {' '.join([c.name for c in conditions])} starting_point={starting_point}"
                )

                solver = Solver(stop_conditions=conditions)
                solver.step_size = step_size
                res = solver.solve(f, starting_point, domain, log=True)
                print(res)
                if res.f_value < best.f_value:
                    best = res
                if res.f_value > worst.f_value:
                    worst = res

                min_f_list_local.append(res.f_value)
                i += 1
            min_f_list.append(min(min_f_list_local))
            if len(conditions_list) == 1:
                #  plotting needs at least 2 conditions. It simulates 2 conditions
                min_f_list.append(min(min_f_list_local))

    condition_name_list = [
        [c.name for c in conditions] for conditions in conditions_list
    ]

    if len(conditions_list) == 1:
        #  plotting needs at least 2 conditions. It simulates 2 conditions
        condition_name_list.append(condition_name_list[0])

    plot_experiment_results(min_f_list, condition_name_list, step_sizes)

    plot_single_result(best, f, domain, title="Best result")
    plot_single_result(worst, f, domain, title="Worst result")
