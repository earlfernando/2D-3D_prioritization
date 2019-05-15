a = [1 , 2 , 3 ,5,6 ,7]
b= [10,11,12,13,14,17,18,19,20,7]
iterator_1= iter(a)
iterator_2 = iter(b)
end_1 = a[-1]
end_2 = b[-1]
start_1 = next(iterator_1)
start_2 = next(iterator_2)
sols_tmep =[]
while start_1 != end_1 and start_2 != end_2:
    if start_1 <= start_2:
        sols_tmep.append(start_1)
        start_1 = next(iterator_1)
    else:
        sols_tmep.append(start_2)
        start_2 = next(iterator_2)
print(sols_tmep)
for i in iterator_2:
    sols_tmep.append(i)
for i in iterator_1:
    sols_tmep.append(i)
print(sols_tmep)

from itertools import groupby
import  numpy as np
from operator import itemgetter

things = [[1, "bear"], [1, "duck"], [2, "cactus"], [3, "speed boat"], [3, "school bus"]]
print(sorted(things,key=lambda x: x[0], reverse= True))
print(sorted(things, key=itemgetter(1)))
for key, group in groupby(things, lambda x: x[0]):
    print(group,key)
    for thing in group:

        print( "A %s is a %s." % (thing[1], key))

for i in range(1,10):
    print(i)