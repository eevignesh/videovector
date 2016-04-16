import os

tmpfile = '/scr/r6/vigneshr/tmp.txt'
os.system('nvidia-smi > ' + tmpfile)
lines = open(tmpfile, 'r').read().splitlines()

qq = [x for x in lines if 'MiB' in x and not 'Default' in x]
pids = [x.split()[2] for x in qq]
gpus = [x.split()[1] for x in qq]
mem = [x.split()[-2] for x in qq]

used = []
for ix in [6,5,4,3,2,1,0]:
  li = [i for i in range(len(gpus)) if int(gpus[i]) == ix]
  print 'gpu %d:' %(ix, ) 
  pidi = [pids[i] for i in li]
  for j,p in enumerate(pidi):
    if p in used: continue

    os.system('ps aux | grep ' + p + ' | grep -v grep > ' + tmpfile)
    pp = open(tmpfile, 'r').read()
    user = pp.split()[0]
    print 'used by %s with pid %s, using %s' % (user, p, mem[li[j]])

  used.extend(pidi)



# for q in qq:
#   os.system('ps aux | grep ' + q + ' | grep -v grep > ' + tmpfile)
#   p = open(tmpfile, 'r').read()
#   user = p.split()[0]
#   print q, user




