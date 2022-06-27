#!/usr/bin/env conda-env-py3

import pandas as pd
import tangos
import tangos.examples.mergers as mergers

import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
h = 0.6776942783267969

from tangos_halo_module.halo_properties import ID_to_sim_halo_snap, get_main_progenitor_branch, impact_parameter, infall_final_n_particles, infall_final_coordinates, apocentric_distance, disruption_time, accretion_time, orbit_interpolation, infall_velocity, quenching_time, max_sSFR_time, max_mass_time 
from tangos_halo_module.path import get_file_path, get_halo_snap_num
from tangos_halo_module.halos import ID_to_tangos_halo, get_survivors, get_zombies, get_host, get_survivor_IDs, get_zombie_IDs, blockPrint, enablePrint

def get_dataframe(ID, sim=0, status=0, resolution=1000):
    if ID != 0:
        sim, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
    sat = ID_to_tangos_halo(ID=ID, resolution=resolution)

    infall_mass=[]; time_infall=[]; infall_mass_ratio=[]; infall_ID=[]; time_quench=[]; max_sSFR=[]; time_max_sSFR=[]
    time_disruption=[]; infall_relative_velocity=[]; infall_radial_velocity=[]; infall_tangential_velocity=[]
    max_stellarmass=[]; time_max_stellarmass=[]
    time_closest_approch=[]; distance_closest_approach_hostRvir=[]; distance_closest_approach_pkpc=[]
    infall_x=[]; infall_y=[]; infall_z=[]; infall_distance=[]; infall_phi_z=[]; infall_z_away=[]
    final_x=[]; final_y=[]; final_z=[]; final_distance=[]; final_phi_z=[]; final_z_away=[]
    infall_n_dm=[]; infall_n_star=[]; infall_n_gas=[]; final_n_dm=[]; final_n_star=[]; final_n_gas=[]

    impact_parameter_kpc=[]; impact_parameter_hostRvir=[]; time_impact=[]
    
    satinfall_time, satinfall_mass, satinfall_mass_ratio, satinfall_id, z = accretion_time(ID=ID, simulation=sim, status=status, 
                                                                                           tangos_halo=sat, halo_id=0, 
                                                                                           snap_num=0, category='stars', 
                                                                                           resolution=resolution)
    infall_mass.append(satinfall_mass)
    time_infall.append(satinfall_time)
    infall_mass_ratio.append(satinfall_mass_ratio)
    infall_ID.append(satinfall_id)

    satquenching_time, x, y = quenching_time(ID=ID, simulation=sim, status=status, 
                                       tangos_halo=sat, halo_id=0, 
                                       snap_num=0, threshold=1e-11, resolution=resolution)
    time_quench.append(satquenching_time)

#     max_satsSFR, max_satsSFR_time = max_sSFR_time(ID=ID, simulation=sim, status=status, 
#                                                   tangos_halo=sat, halo_id=0, snap_num=0, 
#                                                   resolution=resolution)
#     max_sSFR.append(max_satsSFR)
#     time_max_sSFR.append(max_satsSFR_time)

    satdisruption_time = disruption_time(ID=ID, simulation=sim, status=status, 
                                         tangos_halo=sat, halo_id=0, snap_num=0, resolution=resolution)
    time_disruption.append(satdisruption_time)

    relvel, radvel, tanvel, x = infall_velocity(ID=ID, simulation=sim, status=status, 
                                                tangos_halo=sat, halo_id=0, snap_num=0, resolution=resolution)
    infall_relative_velocity.append(relvel)
    infall_radial_velocity.append(radvel)
    infall_tangential_velocity.append(tanvel)

    max_satM, max_satM_time = max_mass_time(ID=ID, simulation=sim, status=status, 
                                            tangos_halo=sat, halo_id=0, snap_num=0, category='stars', 
                                            resolution=resolution)
    max_stellarmass.append(max_satM)
    time_max_stellarmass.append(max_satM_time)

    time_of_closest_approach, Rvir_distance, min_lin_distance = apocentric_distance(ID=ID, simulation=sim, 
                                                                                    status=status, 
                                                                                    tangos_halo=sat, 
                                                                                    halo_id=0, snap_num=0, 
                                                                                    resolution=resolution)
    time_closest_approch.append(time_of_closest_approach)
    distance_closest_approach_hostRvir.append(Rvir_distance) 
    distance_closest_approach_pkpc.append(min_lin_distance)

    (xi, yi, zi, Ri, Phii, awayi), (xf, yf, zf, Rf, Phif, awayf) = infall_final_coordinates(ID=ID, simulation=sim, 
                                                                                            status=status, 
                                                                                            tangos_halo=sat, 
                                                                                            halo_id=0, 
                                                                                            snap_num=0, 
                                                                                            resolution=resolution)
    infall_x.append(xi)
    infall_y.append(yi)
    infall_z.append(zi) 
    infall_distance.append(Ri)
    infall_phi_z.append(Phii)
    infall_z_away.append(awayi)
    final_x.append(xf) 
    final_y.append(yf) 
    final_z.append(zf) 
    final_distance.append(Rf)
    final_phi_z.append(Phif) 
    final_z_away.append(awayf)

    (gi, si, dmi), (gf, sf, dmf) = infall_final_n_particles(ID=ID, simulation=sim, status=status, tangos_halo=sat, halo_id=0, snap_num=0, 
                                                            resolution=resolution)
    infall_n_dm.append(dmi)
    infall_n_star.append(si) 
    infall_n_gas.append(gi) 
    final_n_dm.append(dmf) 
    final_n_star.append(sf)
    final_n_gas.append(gf)
    
    d_impact, d_impact_Rvir, t_impact = impact_parameter(ID=ID, simulation=sim, status=status, tangos_halo=sat, 
                                                         halo_id=0, snap_num=0, resolution=resolution)
    impact_parameter_kpc.append(d_impact) 
    impact_parameter_hostRvir.append(d_impact_Rvir)
    time_impact.append(t_impact)

    d = {'ID': ID, 
         'Simulation': sim,
         'Status': status,
         'infall_mass': infall_mass, 
         'time_infall': time_infall,
         'infall_mass_ratio': infall_mass_ratio,
         'infall_ID': infall_ID,
         'time_quench': time_quench,
#          'max_sSFR': max_sSFR,
#          'time_max_sSFR': time_max_sSFR,
         'time_disruption': time_disruption,
         'infall_relative_velocity': infall_relative_velocity,
         'infall_radial_velocity': infall_radial_velocity,
         'infall_tangential_velocity': infall_tangential_velocity,
         'max_stellarmass': max_stellarmass,
         'time_max_stellarmass': time_max_stellarmass,
         'time_closest_approch': time_closest_approch,
         'distance_closest_approach_hostRvir': distance_closest_approach_hostRvir,
         'distance_closest_approach_pkpc': distance_closest_approach_pkpc,
         'infall_x': infall_x,
         'infall_y': infall_y,
         'infall_z': infall_z,
         'infall_distance': infall_distance,
         'infall_phi_z': infall_phi_z,
         'infall_z_away': infall_z_away,
         'final_x': final_x,
         'final_y': final_y,
         'final_z': final_z,
         'final_distance': final_distance, 
         'final_phi_z': final_phi_z, 
         'final_z_away': final_z_away,
         'infall_n_dm': infall_n_dm, 
         'infall_n_star': infall_n_star, 
         'infall_n_gas': infall_n_gas, 
         'final_n_dm': final_n_dm, 
         'final_n_star': final_n_star, 
         'final_n_gas': final_n_gas, 
         'impact_parameter_kpc': impact_parameter_kpc, 
         'impact_parameter_hostRvir': impact_parameter_hostRvir, 
         'time_impact': time_impact,}

    df = pd.DataFrame(data=d)
    return df
# df = get_dataframe(ID=148409612, resolution='Mint')

# Initialize dataframe
column_names = ['ID', 'Simulation',
                'Status', 'infall_mass', 
                'time_infall', 'infall_mass_ratio', 'infall_ID',
                'time_quench', 'max_sSFR',
                'time_max_sSFR', 'time_disruption',
                'infall_relative_velocity', 'infall_radial_velocity',
                'infall_tangential_velocity', 'max_stellarmass',
                'time_max_stellarmass', 'time_closest_approch',
                'distance_closest_approach_hostRvir', 'distance_closest_approach_pkpc',
                'infall_x', 'infall_y',
                'infall_z', 'infall_distance',
                'infall_phi_z', 'infall_z_away',
                'final_x', 'final_y',
                'final_z', 'final_distance', 
                'final_phi_z', 'final_z_away',
                'infall_n_dm', 'infall_n_star', 
                'infall_n_gas', 'final_n_dm', 
                'final_n_star', 'final_n_gas']
df = pd.DataFrame(columns = column_names)

for sim in ['h148', 'h229', 'h242', 'h329']:
    print(sim)
    for status in ['Survivor', 'Zombie']:
        print(status)
        if status=='Survivor':
            IDs = get_survivor_IDs(simulation=sim, resolution=100)
        elif status=='Zombie':
            IDs = get_zombie_IDs(simulation=sim, resolution=100)
        for ID in IDs:
            print('Started ', ID)
            sat_df = get_dataframe(ID=ID, sim=sim, status=status, resolution=100)
            df = pd.concat([df, sat_df])
            print('Finished ', ID)
            df.to_csv('Data100.csv', index=True)
            print('Saved Data100.csv')
            
print('Finished!!')