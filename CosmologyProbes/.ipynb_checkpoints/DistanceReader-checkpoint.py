import os
import math as m
import statistics as s
from scipy import stats
import scipy.linalg as la
import numpy as np
from scipy.integrate import quad

data_directory='../Data/distanceladder/data/'


def anchor_reader(file,index_array,header_length,delim):

    base_dir = data_directory
    
    #stable method to read in tabular data
    file = open(base_dir + file)
    table = file.readlines()
    file.close()
    
    #defining the length of the actual data of the table
    i = len(table)-header_length
    
    #preallocating the arrays for memory
    host_list = []
    dist_np = np.zeros(i)
    ddist_np = np.zeros(i)
    
    lc = 0
    
    for line in table:
    
        lc += 1
        i = lc - header_length - 1
        data = line.strip('\n').split(delim)

        if lc>header_length:
            if index_array[0] is not np.NaN:
                host_list.append(data[index_array[0]])
            else:
                host_list.append(np.NaN)
        
            if index_array[1] is not np.NaN:
                dist_np[i] = data[index_array[1]]
            else:
                dist_np[i] = np.NaN
        
            if index_array[2] is not np.NaN:
                ddist_np[i] = data[index_array[2]]
            else:
                ddist_np[i] = np.NaN
        
    name = []
    for a in range(len(host_list)):
        name.append([host_list[a],dist_np[a],ddist_np[a]])
                
    return name




# reads in the SHOES cepheid data
def SHOES_ceph_table_reader(file,index_array,header_length,delim):
    #file(string): location of table wished to be read in
    #index_array(array): array containing the column index of the 
    #following attributes:
    #host,ID,period,V,dV,I,dI,NIR,dNIR,OH
    # if attribute does not occur in table, put -1
    #host: index of host column in table
        # list, strings
    #ID: index of ID column in table
        # list, strings
    #period: index of period column in table
        # np array, float
    # the remainder of entries are the index of said variable in table
        # all are np array's, with float values
    #header_length(int): line number of last row of header
    #name(string):user desired prefix for data
    # name_host, name_ID, etc
    #cc(string): character which splits your data
    
    #All data should be located in this parent folder
    base_dir = data_directory
    
    #stable method to read in tabular data
    file = open(base_dir + file)
    table = file.readlines()
    file.close()
    
    #defining the length of the actual data of the table
    i = len(table)-header_length
    
    #preallocating the arrays for memory
    host_list = []
    ID_list = []
    period_np = np.zeros(i)
    V_np = np.zeros(i)
    dV_np = np.zeros(i)
    I_np = np.zeros(i)
    dI_np = np.zeros(i)
    NIR_np = np.zeros(i)
    dNIR_np = np.zeros(i)
    OH_np = np.zeros(i)
    
    lc = 0
    
    for line in table:
    
        lc += 1
        i = lc - header_length - 1
        data = line.strip('\n').split(delim)

        if lc>header_length:
            if index_array[0] is not np.NaN:
                host_list.append(data[index_array[0]])
            else:
                host_list.append(np.NaN)
        
            if index_array[1] is not np.NaN:
                ID_list.append(data[index_array[1]])
            else:
                ID_list.append(np.NaN)
        
            if index_array[2] is not np.NaN:
                period_np[i] = data[index_array[2]]
            else:
                period_np[i] = np.NaN
        
            if index_array[3] is not np.NaN:
                V_np[i] = data[index_array[3]]
            else:
                V_np[i] = np.NaN
        
            if index_array[4] is not np.NaN:
                dV_np[i] = data[index_array[4]]
            else:
                dV_np[i] = np.NaN
            
            if index_array[5] is not np.NaN:
                I_np[i] = data[index_array[5]]
            else:
                I_np[i] = np.NaN
        
            if index_array[6] is not np.NaN:
                dI_np[i] = data[index_array[6]]
            else:
                dI_np[i] = np.NaN
        
            if index_array[7] is not np.NaN:
                NIR_np[i] = data[index_array[7]]
            else:
                NIR_np[i] = np.NaN
        
            if index_array[8] is not np.NaN:
                dNIR_np[i] = data[index_array[8]]
            else:
                dNIR_np[i] = np.NaN
        
            if index_array[9] is not np.NaN:
                OH_np[i] = data[index_array[9]]
            else:
                OH_np[i] = np.NaN

        name = []
    for a in range(len(host_list)):
        name.append([host_list[a],ID_list[a],period_np[a],V_np[a],dV_np[a],I_np[a],dI_np[a],NIR_np[a],dNIR_np[a],OH_np[a]])
                
    return name



# reads in the LMC data
def LMC_ceph_reader(file,index_array,header_length,delim):
    #file(string): location of table wished to be read in
    #index_array(array): array containing the column index of the following attributes:
    #host,ID,period,V,dV,I,dI,NIR,dNIR,OH
    # if attribute does not occur in table, put -1
    #host: index of host column in table
        # list, strings
    #ID: index of ID column in table
        # list, strings
    #period: index of period column in table
        # np array, float
    # the remainder of entries are the index of said variable in table
        # all are np array's, with float values
    #header_length(int): line number of last row of header
    #name(string):user desired prefix for data
    # name_host, name_ID, etc
    #cc(string): character which splits your data

    #All data should be located in this parent folder
    base_dir = data_directory

    #stable method to read in tabular data
    file = open(base_dir + file)
    table = file.readlines()
    file.close()

    #defining the length of the actual data of the table
    i = len(table)-header_length

    #preallocating the arrays for memory
    host_list = []
    ID_list = []
    period_np = np.zeros(i)
    mHW_np = np.zeros(i)
    dmHW_np = np.zeros(i)

    lc = 0
    for line in table:

        lc += 1
        i = lc - header_length - 1
        data = line.strip('\n').split(delim)
        if lc>header_length:
            if index_array[0] is not np.NaN:
                host_list.append(data[index_array[0]])
            else:
                host_list.append(np.NaN)

            if index_array[1] is not np.NaN:
                ID_list.append(data[index_array[1]])
            else:
                ID_list.append(np.NaN)

            if index_array[2] is not np.NaN:
                period_np[i] = data[index_array[2]]
            else:
                period_np[i] = np.NaN

            if index_array[3] is not np.NaN:
                mHW_np[i] = data[index_array[3]]
            else:
                mHW_np[i] = np.NaN

            if index_array[4] is not np.NaN:
                dmHW_np[i] = data[index_array[4]]
            else:
                dmHW_np[i] = np.NaN

    name = []
    for a in range(len(host_list)):
        if period_np[a] > .8:
            name.append(['LMC',ID_list[a],period_np[a],mHW_np[a],dmHW_np[a]])
                
    return name


# reads in cepheid calibrated SN data
def SHOES_sn_table_reader(file,index_array,header_length,delim):
    
    #All data should be located in this parent folder
    base_dir = data_directory
    
    #stable method to read in tabular data
    file = open(base_dir + file)
    table = file.readlines()
    file.close()
    
    #defining the length of the actual data of the table
    i = len(table)-header_length
    
    #preallocating the arrays for memory
    host_list = []
    ID_list = []
    m_np = np.zeros(i)
    dm_np = np.zeros(i)
    
    lc = 0
    
    for line in table:
    
        lc += 1
        i = lc - header_length - 1
        data = line.strip('\n').split(delim)

        if lc>header_length:
        
            if index_array[0] is not np.NaN:
                host_list.append(data[index_array[0]])
            else:
                host_list.append(np.NaN)
        
            if index_array[1] is not np.NaN:
                ID_list.append(data[index_array[1]])
            else:
                ID_list.append(np.NaN)
        
            if index_array[2] is not np.NaN:
                m_np[i] = data[index_array[2]]
            else:
                m_np[i] = np.NaN
        
            if index_array[3] is not np.NaN:
                dm_np[i] = data[index_array[3]]
            else:
                dm_np[i] = np.NaN
        
    name = []
    for a in range(len(host_list)):
        name.append([host_list[a],ID_list[a],m_np[a],dm_np[a]])
                
    return name



#reads in the pantheon hubble flow supernova data
def panth_sn_table_reader(file,index_array,header_length,delim):

    #All data should be located in this parent folder
    base_dir = data_directory

    #stable method to read in tabular data
    file = open(base_dir + file)
    table = file.readlines()
    file.close()

    #defining the length of the actual data of the table
    i = len(table)-header_length

    #preallocating the arrays for memory
    ID_list = []
    z_hel_np = np.zeros(i)
    z_cmb_np = np.zeros(i)
    dz_cmb_np = np.zeros(i)
    m_np = np.zeros(i)
    dm_np = np.zeros(i)

    lc = 0

    for line in table:

        lc += 1
        i = lc - header_length - 1
        data = line.strip('\n').split(delim)

        if lc>header_length:

            if index_array[0] is not np.NaN:
                ID_list.append(data[index_array[0]])
            else:
                ID_list.append(np.NaN)

            if index_array[1] is not np.NaN:
                z_hel_np[i] = data[index_array[1]]
            else:
                z_hel_np[i] = np.NaN

            if index_array[2] is not np.NaN:
                z_cmb_np[i] = data[index_array[2]]
            else:
                z_cmb_np[i] = np.NaN

            if index_array[3] is not np.NaN:
                dz_cmb_np[i] = data[index_array[3]]
            else:
                dz_cmb_np[i] = np.NaN

            if index_array[4] is not np.NaN:
                m_np[i] = data[index_array[4]]
            else:
                m_np[i] = np.NaN

            if index_array[5] is not np.NaN:
                dm_np[i] = data[index_array[5]]
            else:
                dm_np[i] = np.NaN
    
    
    name = []
    for a in range(len(ID_list)):
        if z_cmb_np[a] > 0.0023:
            #if z_cmb_np[a] < 0.15:
            name.append([ID_list[a],z_hel_np[a],z_cmb_np[a],dz_cmb_np[a],m_np[a],dm_np[a]])
    return name