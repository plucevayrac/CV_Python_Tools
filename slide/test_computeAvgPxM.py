import math


mesures = {
    ((541,553),(581,100),68.91),
    ((634,615),(337,568),24.93),
    ((490,173),(633,614),69.81),
    ((478,721),(483,213),80.31),
    ((949,180),(486,718),119.29),
    ((248,263),(706,263),40.),
    ((542,552),(580,511),68.61),
    ((488,714),(483,406),48.74),
    ((417,710),(651,224),92.09)
}

def euclideanDist(p1, p2, verbose=False):
    if verbose:
        print('**** Entering EuclideanDist ****')
        print('p1: {}x{} p2: {}x{}'.format(p1[0],p1[1],p2[0],p2[1]))
        
    ret = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return ret

for mes in mesures :
    print('**** **** Mesure: {} **** ****'.format(mes))
    p1, p2, dist = mes
    print('Dist in meters = {}'.format(dist))
    eucliDist = euclideanDist(p1, p2)
    print('Euclidean Dist = {}'.format(eucliDist))
    pix_ratio_MpP = dist / eucliDist
    print('Pix Ratio (M/P) = {}'.format(pix_ratio_MpP))
