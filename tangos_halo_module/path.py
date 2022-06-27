import re

def ID_to_sim_halo_snap(ID=0):
    '''
    input params: 
        ID: new halo ID of the format SSSXXXXYYYY [3# sim] 
                                                 +[4# latest snap num when halo exists] 
                                                 +[1-4# sim halo ID at latest snap]
    output params: 
       simulation: corresponding simulation in format 'h329'
       status: status of halo in format 'Host', 'Survivor' or 'Zombie'
       halo_id: id of halo in format '1', '283', etc.
       snap_num: final snapshot number in format '0071', '4096', etc.
    '''
    simulation = 'h'+str(ID)[:3] #eg. h329        
    snap_num = str(str(ID)[3:7])
    halo_id = str(ID)[7:]
    
    #Status:
    if snap_num == '4096':
        if halo_id == '1':
            status='Host'
        else:
            status='Survivor'
    else:
        status='Zombie'
        
    return simulation, status, halo_id, snap_num



def get_file_path(ID=0, tangos_halo=0, simulation=0, status=0, halo_id=0, snap_num=0, resolution=1000):
    '''
    input params: 
        tangos_halo: a valid Tangos halo object
        simulation: h148, h229, h242, h329: string
        status: 'Host', 'Survivor', 'Zombie'
        halo_id: halo id: string or numeric
        snap_num: 4 digit simulation snapshot number: string
    output params:
        path: path to local halo data file: string
    '''
    import sys

    if ID:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
        ID=str(ID)
#         simulation = ID[:3]
        snapnum = 'snap_' + ID[3:7]
        halo_num = 'halo_' + ID[7:]
    elif tangos_halo:
        halo = str(tangos_halo)
        halo_num = re.findall(r'halo_+[\d]+', halo)[0] #'halo_123'
        snapnum = 'snap_' + re.findall(r'.00+[\d]+', halo)[0][3:] #'snap_0123' 'snap_0012'
    elif halo_id and snap_num:
        halo_num = 'halo_' + str(halo_id)
        snapnum = 'snap_' + str(snap_num)
    else:
        raise ValueError("Halo %r not found. Try again with additional information. You got this tho." % (halo))
    if resolution==1000:
        return f'Halo_Files/'+ str(simulation) +'/'+ str(status) +'/'+ str(snapnum) +'_'+ str(halo_num) +'.hdf5'
    elif resolution==100:
        return f'Halo_Files/'+ str(simulation) +'_100/'+ str(status) +'/'+ str(snapnum) +'_'+ str(halo_num) +'.hdf5'
    elif resolution=='Mint':
        return f'Mint_Data/Halo_Files/'+ str(simulation) +'/'+ str(status) +'/'+ str(snapnum) +'_'+ str(halo_num) +'.hdf5'
    else:
        raise ValueError("Resolution not implemented yet.")    
    



def get_halo_snap_num(tangos_halo):
    '''
    input params: 
        tangos_halo: a valid Tangos halo object
        simulation: h148, h229, h242, h329: string
    output params:
        [0] halo_num: halo number
        [1] snap_num: snapshot number
        [2] new_id: new halo ID of the format SSSXXXXYYYY [3# sim] 
                                                     +[1-4# latest snap num when halo exists] 
                                                     +[4# sim halo ID at latest snap]
    '''
    import re
    halo = str(tangos_halo)
    sim = re.findall(r'snapshots/h+[\d]+', halo)[0][11:]
    halo_num = re.findall(r'halo_+[\d]+', halo)[0][5:] #'123'
    snap_num = re.findall(r'.00+[\d]+', halo)[0][3:] #'0123' '0012'
    new_id = str(sim) + str(snap_num) + str(halo_num) #'329_4096_1'
    simname = 'h'+str(sim)
    return halo_num, snap_num, new_id, simname

def read_file(path, halo_num):
    '''
    Function to read in the timestep bulk-processing datafile (from /home/akinhol/Data/Timescales/DataFiles/{name}.data)
    ^^^ This function is from Hollis ^^^
    '''
    import pickle
    import pandas as pd
    data = []
    with open(path,'rb') as f:
        while True:
            try: 
                data.append(pickle.load(f))
            except EOFError:
                break
    data = pd.DataFrame(data)
    
    if not halo_num=='all':
        data = data[data.z0haloid == halo_num]

    return data
