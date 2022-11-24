string1 = 'http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/sub1/nres12/allskymap_nres12r'

for i in range(108):
    for j in range(1,67):
        if i < 10:
            direccion = string1 + '00' + str(i) + '.zs' +  str(j) +'.mag.dat'
        if 10 <= i <= 99:
            direccion = string1 + '0' + str(i) + '.zs' +  str(j) +'.mag.dat'
        if 99 < i:
            direccion = string1 +  str(i) + '.zs' +  str(j) +'.mag.dat'
