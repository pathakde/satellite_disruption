#!/usr/bin/env python

import pandas as pd
import tangos
import tangos.examples.mergers as mergers
import pynbody

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
h = 0.6776942783267969

from tangos_halo_module.halo_properties import track_halo_property, get_timesteps, ID_to_sim_halo_snap, infall_final_n_particles, infall_final_coordinates, apocentric_distance, disruption_time, accretion_time, orbit_interpolation, infall_velocity, quenching_time, max_sSFR_time, max_mass_time 
from tangos_halo_module.path import get_file_path, get_halo_snap_num, read_file
from tangos_halo_module.halos import ID_to_tangos_halo, get_survivors, get_main_progenitor_branch, get_zombies, get_host, get_survivor_IDs, get_zombie_IDs, blockPrint, enablePrint, tangos_to_pynbody_halo

data = pd.read_csv('Data100.csv')
ids = np.asarray(data['ID'])
simul = np.asarray(data['Simulation'])

# Initialize DFs
for sim in ['h329', 'h242', 'h229', 'h148']:
    idx = ids[simul==sim]
    column_names = ['snapshot']+list(idx)
    df = pd.DataFrame(columns = column_names, dtype=str)#.apply(str)

    df.to_csv(str(sim)+'MPB_CenterCoords.csv', index=False)
    df.to_csv(str(sim)+'MPB_RSmass.csv', index=False)
    df.to_csv(str(sim)+'MPB_RDmass.csv', index=False)
    ts = get_timesteps(simulation=sim, resolution=100)[3]
    df['snapshot'] = ts
    df.to_csv(str(sim)+'MPB_CenterCoords.csv', index=False)
    df.to_csv(str(sim)+'MPB_RSmass.csv', index=False)
    df.to_csv(str(sim)+'MPB_RDmass.csv', index=False)
    print(str(sim)+'MPB_CenterCoords.csv initialized.')
    print(str(sim)+'MPB_RSmass.csv initialized.')
    print(str(sim)+'MPB_RDmass.csv initialized.')

# Get data
for sim in ['h329', 'h242', 'h229', 'h148']:
    print('started simulation ', sim) 
    df = pd.read_csv(str(sim)+'MPB_ids.csv')
    IDs = np.asarray(df.columns)[1:]
    print('my IDs are ', IDs)
    ts = get_timesteps(simulation=sim, resolution=100)[3]
    print('the timesteps are ', ts)
    
    #Read in new dfs
    df_RSmass = pd.read_csv(str(sim)+'MPB_RSmass.csv')
    df_RDmass = pd.read_csv(str(sim)+'MPB_RDmass.csv')
    df_CenterCoords = pd.read_csv(str(sim)+'MPB_CenterCoords.csv')
    
    
    #Iterate through each timestep
    for index in range(0, len(ts)):
        #index is #row
        snapnum = ts[index]
        print('starting with timestep ', snapnum)
        
        halo_ids = np.asarray(df.loc[index][1:])
        print('for timestep ', snapnum, ' my halo IDs are ', halo_ids)
    
        if sim == 'h148':
            snapshot = pynbody.load("/home/pathakde/Sims/"+ str(sim) + ".cosmo50PLK.3072g/"
                                    + str(sim) + ".cosmo50PLK.3072g3HbwK1BH/snapshots/"
                                    + str(sim) + ".cosmo50PLK.3072g3HbwK1BH.00"+ str(snapnum))
        else:
            snapshot = pynbody.load("/home/pathakde/Sims/"+ str(sim) + ".cosmo50PLK.3072g/"
                                    + str(sim) + ".cosmo50PLK.3072gst5HbwK1BH/snapshots/"
                                    + str(sim) + ".cosmo50PLK.3072gst5HbwK1BH.00"+ str(snapnum))
        snapshot.physical_units()
        all_halos = snapshot.halos()
        print(snapshot)
        print(all_halos)
        
        count=0 #This is #column-1
        
        #Iterate through each halo
        for idx in halo_ids:
            print('started [', index, ', ', count, ']')
            ID = IDs[count]
            index=int(index)
            count=int(count)
            ID=int(ID)
            print('my ID is ', ID)
            print('')
            if idx==0:
                #set everything to -1 and move on
                df_CenterCoords.at[index, ID] = '[-1, -1, -1, -1]'
                df_RSmass.at[index, ID] = '[-1, -1, -1, -1]'
                df_RDmass.at[index, ID] = '[-1, -1, -1, -1]'
            else:
                idx=int(idx)
                simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=idx)
                halo=all_halos[int(halo_id)]
                print(halo)
                xc, yc, zc = pynbody.analysis.halo.center(halo, mode='ssc', retcen=True, move_all=False)  
                df_CenterCoords.at[index, ID] = str([xc, yc, zc])
                #---------------------------------------Stars-------------------------------------#
                mass = np.asarray(halo.s['mass'])
                if len(mass)<4:
                    percentiles = [0, 0, 0, 0]
                else:
                    R = np.asarray(np.sqrt((halo.s['x']-xc)**2 + 
                                           (halo.s['y']-yc)**2 + 
                                           (halo.s['z']-zc)**2))
                    zipped_lists = zip(R, mass)
                    sorted_pairs = sorted(zipped_lists)
                    tuples = zip(*sorted_pairs)
                    R_sort, mass_sort = [list(tuple) for tuple in  tuples]
                    percentiles=[]
                    for bound in [0.25, 0.50, 0.75, 1.0]:
                        percentile_indices = np.where(np.cumsum(np.array(mass_sort))<=bound*np.sum(np.array(mass_sort)))
                        if len(percentile_indices[0])<1:
                            R_percentile=0
                        else:
                            R_percentile = R_sort[max(percentile_indices[0])] #max([np.array(R_sort)[i] for i in percentile_indices])
                        percentiles.append(R_percentile)

                print('Stellar Mass Percentiles: ', percentiles)
                #add data in dfs
                df_RSmass.at[index, ID] = str(percentiles)
                #---------------------------------------Dark Matter-------------------------------------#
                mass = np.asarray(halo.dm['mass'])
                if len(mass)<4:
                    percentiles = [0, 0, 0, 0]
                else:
                    R = np.asarray(np.sqrt((halo.dm['x']-xc)**2 + 
                                           (halo.dm['y']-yc)**2 + 
                                           (halo.dm['z']-zc)**2))
                    zipped_lists = zip(R, mass)
                    sorted_pairs = sorted(zipped_lists)
                    tuples = zip(*sorted_pairs)
                    R_sort, mass_sort = [list(tuple) for tuple in  tuples]
                    percentiles=[]
                    for bound in [0.25, 0.50, 0.75, 1.0]:
                        percentile_indices = np.where(np.cumsum(np.array(mass_sort))<=bound*np.sum(np.array(mass_sort)))
                        if len(percentile_indices[0])<1:
                            R_percentile=0
                        else:
                            R_percentile = R_sort[max(percentile_indices[0])] #max([np.array(R_sort)[i] for i in percentile_indices])
                        percentiles.append(R_percentile)
                print('Dark Matter Mass Percentiles: ', percentiles)
                #add data in dfs
                df_RDmass.at[index, ID] = str(percentiles)
#             print(df_CenterCoords)
#             print(df_RDmass)
#             print(df_RSmass)
            print('finished [', index, ', ', count+1, ']') 
            
            df_CenterCoords.to_csv(str(sim)+'MPB_CenterCoords.csv', index=False)
            df_RSmass.to_csv(str(sim)+'MPB_RSmass.csv', index=False)
            df_RDmass.to_csv(str(sim)+'MPB_RDmass.csv', index=False)
            print('saved [', index, ', ', count, ']')
            
            #Move on to next id == next column
            count+=1
        
        print('finished timestep ', snapnum)
    print('finished simulation ', sim)
print('FINALLY WE ARE DONE!!!')

    