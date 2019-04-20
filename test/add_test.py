from src.isdm import MCRVector,Vector

vecs = [ MCRVector.random_vector() for _ in range(10) ]


#cla_mcr_res = addMCR([v._dims for v in vecs], [0,15])
#print(cla_mcr_res)

#isdm_res = sum(vecs)
#print(isdm_res)

problem = [3,7,10,13,1,2,8,4,13,3]

values = [Vector(v) for v in problem]
r = values[0]
for v in range(1,len(values)):
    r = r+values[v]
    print(r.mag, r.theta)
