import numpy as np
import networkx as nx

def print_me(mylist):
    print("[", end=" ")
    for num in mylist:
        print("{:.4f}".format(num), end=" ")
    print("]", end="\n")
# --------------------------------------------

max_iter=20
print("")
print("damping")

p = [1,0,0,0]
p = np.asarray(p)
L = [ [0.05,0.85,0.05,0.05],
     [0.05,0.05,0.45,0.45],
     [0.05,0.05,0.05,0.85],
     [0.85,0.05,0.05,0.05]]
for i in range(max_iter):
    p = p @ L
    print_me (p)
# --------------------------------------------

print("")
print("three node with loop No damping factor")

max_iter=30
p = [1,0,0]
p = np.asarray(p)
L = [ [0.0,1,0.0],
     [0.0,0.0,1],
     [0.0,1,0.0]]
for i in range(max_iter):
    p = p @ L
    print_me (p)

# --------------------------------------------
print("")
print("three node with loop and damping factor")

max_iter=30
p = [1,0,0]
p = np.asarray(p)
L = [ [0.0667,0.8667,0.0667],
     [0.0667,0.0667,0.8667],
     [0.0667,0.8667,0.0667]]
for i in range(max_iter):
    p = p @ L
    print_me (p)
# --------------------------------------------

print("")
print("handle dangling + damping")

max_iter=20
p = [1,0,0,0]
p = np.asarray(p)
L = [ [0.05,0.85,0.05,0.05],
     [0.05,0.05,0.45,0.45],
     [0.05,0.05,0.05,0.85],
     [0.25,0.25,0.25,0.25]]
for i in range(max_iter):
    p = p @ L
    print_me (p)

#2024-09-09 only test
print("")
print("NO handle dangling + damping")

max_iter=20
p = [1,0,0,0]
p = np.asarray(p)
L = [ [0.05,0.85,0.05,0.05],
     [0.05,0.05,0.45,0.45],
     [0.05,0.05,0.05,0.85],
     [0.05,0.05,0.05,0.05]]
for i in range(max_iter):
    p = p @ L
    print_me (p)
# --------------------------------------------


print("")
print("personalized [1,0,0,0]")
# personalized page rank
p = [1,0,0,0]
p = np.asarray(p)
p0 = p
alpha = 0.85
max_iter = 25
L = [ [0,1,0,0],
     [0,0,0.5,0.5],
     [0,0,0,1],
     [1,0,0,0]]
for i in range(max_iter):
    p = alpha *p @ L + (1-alpha)*p0
    print_me (p)
# --------------------------------------------

print("")
print("personalized [0,1,0,0]")
# personalized page rank
p = [0,1,0,0]
p = np.asarray(p)
p0 = p
alpha = 0.85
max_iter = 25
L = [ [0,1,0,0],
     [0,0,0.5,0.5],
     [0,0,0,1],
     [1,0,0,0]]
for i in range(max_iter):
    p = alpha *p @ L + (1-alpha)*p0
    print_me (p)
# --------------------------------------------

print("")
print("personalized [0.5,0.5,0,0]")
# personalized page rank
p = [0.5,0.5,0,0]
p = np.asarray(p)
p0 = p
alpha = 0.85
max_iter = 25
L = [ [0,1,0,0],
     [0,0,0.5,0.5],
     [0,0,0,1],
     [1,0,0,0]]
for i in range(max_iter):
    p = alpha *p @ L + (1-alpha)*p0
    print_me (p)
    