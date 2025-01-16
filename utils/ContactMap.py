import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import DataHelper as DH

#define coordinate extraction function for PDB
def Get_Ca_Coords(path, pdb='xx', chain='A', check_resi=1, check_length=2000):
    with open(path, mode="r") as file:
        lines = file.readlines()

    out = []
    flag = 0
    for line in lines:
        if line.startswith('ATOM') and line.split()[4][0] == 'A' and line.split()[2] == 'CA':

            if len(line.split()[4]) == check_resi and flag < check_length:
                flag = flag+1

                resi = line.split()[5]
                resn = line.split()[3]
                x = line.split()[6]
                y = line.split()[7]
                z = line.split()[8]
                out.append([resi, resn, x, y, z])

            elif flag < check_length:
                flag = flag+1

                resi = line.split()[4][1:5]
                resn = line.split()[3]
                x = line.split()[5]
                y = line.split()[6]
                z = line.split()[7]
                out.append([resi, resn, x, y, z])

    df = pd.DataFrame(out, columns=['res_num', 'res_name', 'x', 'y', 'z'])
    return df


#define coordinate extraction function for AF2
def Get_Ca_Coords_AF2(path, AF2='xx', chain='A', check_resi=1, pLDDT_cutoff=70, check_length=2000):
    with open(path, mode="r") as file:
        lines = file.readlines()

    out = []
    flag = 0
    for line in lines:
        if line.startswith('ATOM') and line.split()[4][0] == 'A' and line.split()[2] == 'CA':
            #print(line.split()[9])
            if len(line.split()[4]) == check_resi and float(line.split()[10]) > pLDDT_cutoff \
               and flag < check_length:
                flag = flag+1

                resi = line.split()[5]
                resn = line.split()[3]
                x = line.split()[6]
                y = line.split()[7]
                z = line.split()[8]
                out.append([resi, resn, x, y, z])

            elif len(line.split()[4]) != check_resi and float(line.split()[9]) > pLDDT_cutoff \
                 and flag < check_length:
                flag = flag+1

                resi = line.split()[4][1:5]
                resn = line.split()[3]
                x = line.split()[5]
                y = line.split()[6]
                z = line.split()[7]
                out.append([resi, resn, x, y, z])

    df = pd.DataFrame(out, columns=['res_num', 'res_name', 'x', 'y', 'z'])
    return df


#calculate distance between two residues(Ca) for a target
def Get_Contact_Distance(path):
    #save ca coordinates
    if '_AF_v2' in path:
        ca_coords = Get_Ca_Coords_AF2(path)
    else:
        ca_coords = Get_Ca_Coords(path)

    #ca_coords.to_csv('ca_coords.csv')
    #pairwise distances
    dist_arr = pairwise_distances(ca_coords[['x', 'y', 'z']].values)
    return dist_arr, ca_coords[['res_name']].values, ca_coords[['x', 'y', 'z']].values


#get adjacency for a target
def Get_Residues_Adjacency(dis_matrix, contact_cutoff=8):
    size = len(dis_matrix)
    adj_matrix = [[0 for col in range(size)] for row in range(size)]

    for i in range(size):
        for j in range(size):
            if dis_matrix[i][j]<contact_cutoff:
                adj_matrix[i][j] = 1

    return adj_matrix


#draw distance map and contact map
def Draw_Dmap(dist_arr):

    fig_dis = plt.figure(figsize=(6, 5))
    p = plt.imshow(dist_arr, cmap='viridis_r', vmax=24, vmin=0)
    plt.colorbar(p, label='Distance ($\mathrm{\AA}$)')
    plt.xlabel('Residue i')
    plt.ylabel('Residue j')
    plt.title('Distance Map1')
    plt.savefig('Distance Map.png', bbox_inches='tight', dpi=330)

def Draw_Cmap(dist_arr):

    fig_con = plt.figure(figsize=(6, 5))
    p = plt.imshow(dist_arr<8, cmap='viridis_r', vmax=1, vmin=0)
    plt.colorbar(p, label='Contact')
    plt.xlabel('Residue i')
    plt.ylabel('Residue j')
    plt.title('Contact Map1')
    plt.savefig('Contact Map.png', bbox_inches='tight', dpi=330)


files_path = "D:/code project/AiPROTACs/data/For DeepPROTACs PROTAC-DB 2.0/e3 ligase pocket 3Dtxt"
files_list = os.listdir(files_path)
files_list.sort(key=lambda x:int(x.split(' ')[0]))
for i in range(len(files_list)):
    print(i)
    dist_arr, res_seq, ca_coords = Get_Contact_Distance(files_path+'/'+files_list[i])
    adj_matrix = Get_Residues_Adjacency(dist_arr)

    #Draw_Dmap(dist_arr)
    #Draw_Cmap(dist_arr)
    np.savetxt('{}.txt'.format(i), dist_arr)
    #np.savetxt('{}.txt'.format(i), res_seq, fmt='%s')
    #np.savetxt('{}.txt'.format(i), ca_coords, fmt='%s')
    #np.savetxt('{}.txt'.format(i), adj_matrix)


'''dist_arr, res_seq, ca_coords = Get_Contact_Distance(files_path+'/'+files_list[231])
adj_matrix = Get_Residues_Adjacency(dist_arr)

#Draw_Dmap(dist_arr)
#Draw_Cmap(dist_arr)
np.savetxt('{}.txt'.format(231), dist_arr)
#np.savetxt('{}.txt'.format(6), res_seq, fmt='%s')
#np.savetxt('{}.txt'.format(6), ca_coords, fmt='%s')
#np.savetxt('{}.txt'.format(6), adj_matrix)'''
