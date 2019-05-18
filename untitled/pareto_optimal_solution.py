import random
from operator import itemgetter
from fptas import FPTAS
from dynamic_programming import dynamic_programming
from ratio_greedy import ratio_greedy
pareto_optimal_solution = []
pareto_optimal_old =[]
pareto_optimal_dub =[]
points =[]
array_for_knapsack =[]
a=[]



def greedy_mine(N, capacity, weight_cost):
    # input cost,prob
    ratios = [(index, item[1] / float(item[0])) for index, item in enumerate(weight_cost)]
    ratios = sorted(ratios, key=lambda x: x[1], reverse=True)
    best_comb = []

    best_cost = 0
    for i in range(N):
        index = ratios[i][0]

        if best_cost+ weight_cost[index][0] <=capacity:
            best_comb.append(index)
            best_value = i+1
            best_cost += weight_cost[index][0]

    return best_cost,best_comb,best_value
val =[]
weight =[]
for i in range (100):
    prob = random.uniform(0,1)
    cost= random.uniform(0,100)
    a.append(cost)
    val.append(prob)
    weight.append(cost)

    points.append([prob,cost])
    array_for_knapsack.append((cost,prob))


max_cost = 500
best_comb,fptas_comb = FPTAS(number= 100, capacity=max_cost,weight_cost=array_for_knapsack)
best_greedy,comb,best_value = greedy_mine(100,capacity= max_cost,weight_cost=array_for_knapsack)
print(fptas_comb)
val = [60, 100, 120]
wt = [10, 20, 30]
W = 50
sum =[(10,60),(20,100),(30,120)]
best_comb,fptas_comb = FPTAS(number=3,capacity=W,weight_cost=sum,scaling_factor=2)
print(best_comb,fptas_comb)

print(best_greedy,best_comb,best_value)

print(comb,best_greedy,best_value)
number_of_points = len(points)
for i in range(number_of_points):

    cost = points[i][1]
    prob = points[i][0]
    if prob == 0.0 or cost >= 200:
        print(cost)

        continue
    else:
        pareto_optimal_old = pareto_optimal_solution
        pareto_optimal_dub = []
        old_number = len(pareto_optimal_solution)
        pareto_optimal_dub.append([prob, cost])
        for k in range(old_number):
            if pareto_optimal_solution[k][1] <= max_cost:
                old_prob = prob + pareto_optimal_solution[k][0]
                old_cost = cost + pareto_optimal_solution[k][1]
                pareto_optimal_dub.append([old_prob, old_cost])
        pareto_optimal_solution = []
        sol_temp = pareto_optimal_dub + pareto_optimal_old
        sol_temp = sorted(sol_temp, key=itemgetter(1))
        last_index = 0
        if len(sol_temp) >= 1:
            for i in range(1, len(sol_temp)):
                last_item = sol_temp[last_index]
                present_item = sol_temp[i]
                if last_item[0] < present_item[0]:
                    if last_item[1] < present_item[1]:
                        pareto_optimal_solution.append(last_item)
                        last_index = i
        if last_index < len(sol_temp):
            pareto_optimal_solution.append(sol_temp[last_index])
print(pareto_optimal_solution)

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
print(pareto_optimal_solution[-1][1])



