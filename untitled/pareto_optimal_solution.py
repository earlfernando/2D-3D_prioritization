import random
from operator import itemgetter
from fptas import FPTAS
from ratio_greedy import ratio_greedy
pareto_optimal_solution = []
pareto_optimal_old =[]
pareto_optimal_dub =[]
points =[]
array_for_knapsack =[]
a=[]
for i in range (100):
    prob = random.uniform(0,1)
    cost= random.uniform(0,100)
    a.append(cost)
    points.append([prob,cost])
    array_for_knapsack.append((prob,cost))


max_cost = 500
best_comb,_ = FPTAS(number= 100, capacity=max_cost,weight_cost=array_for_knapsack,scaling_factor=100)
best_greedy,_ = ratio_greedy(number=100,capacity= max_cost,weight_cost=array_for_knapsack)
print(best_comb,best_greedy)
number_of_points = len(points)
for i in range(number_of_points):

    cost= points[i][1]
    prob = points[i][0]
    if prob==0.0 or cost >=200:
        print(cost)


        continue
    else:
        pareto_optimal_old=pareto_optimal_solution
        pareto_optimal_dub=[]
        old_number = len(pareto_optimal_solution)
        pareto_optimal_dub.append([prob,cost])
        for k in range(old_number):
            if pareto_optimal_solution[k][1] <= max_cost:
                old_prob =prob+ pareto_optimal_solution[k][0]
                old_cost = cost+pareto_optimal_solution[k][1]
                pareto_optimal_dub.append([old_prob,old_cost])
        pareto_optimal_solution =[]
        sol_temp =pareto_optimal_dub + pareto_optimal_old
        sol_temp =sorted(sol_temp, key=itemgetter(1))
        last_index = 0
        if len(sol_temp)>=1:
            for i in range(1,len(sol_temp)):
                last_item = sol_temp[last_index]
                present_item = sol_temp[i]
                if last_item[0]< present_item[0]:
                    if last_item[1]<present_item[1]:
                        pareto_optimal_solution.append(last_item)
                        last_index = i
        if last_index < len(sol_temp):
            pareto_optimal_solution.append(sol_temp[last_index])
print(pareto_optimal_solution)


