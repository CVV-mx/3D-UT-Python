#This program reads in and processes c-scan data in the form of a csv file that
#has been created by UTwin software
#16 September 2024 - Mac Delaney
##
#Functions included in this file:
## read_cscan
## - Reads the raw csv (comma separated value) file, which was converted from a .csc file by the UTwin software
## - Outputs python pickle file, which can be unpacked to retrieve and plot data
##
##
##
## cscan_plots
## - Calculates key outputs like surface contour, plots results. Referenced by both read_cscan and read_processed_cscan
##
## smooth_dent_profile
## - Smooths out dent profile curve, referenced by cscan_plots. Was separated into a new function maily just for code readability
##
## cscan_inspect
## - Once read_cscan or read_processed_cscan have been run, this function allows the user to plot an A-scan curve at a specified location
##
## find_peaks
## - finds peaks in A-scan needed for read_cscan and read_processed_cscan to calculate time of flight (TOF) and amplitude (AMP) of signal
##
## v4 update: change parabolic peak fit to hilbert envelope
## v5 update: clean out unused portions of code, optimize parsing of csv data
## v6 update: implement the pickle format. Process data, save as pickle file, include new function that can read pickle file and re-plot data
##              can then write loop to re-process all data, then plot as needed

#Import libraries that the code will reference
import time
import pickle
from datetime import datetime
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import winsound
from scipy.signal import hilbert
import os
from scipy.ndimage import label
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull
from skimage.measure import find_contours
from sklearn.cluster import DBSCAN


def cscan_plots(pkfile, save_processed=True, tof1plot = 1, tofdplot = 1, dmg = 1, 
                dent = 1, amp1plot = 1, amp2plot = 1, amp3plot = 1, contplot = 1, 
                sect_cut = -1, cont_range = [-1.5,1.5], thk = 'tof', Front = False, 
                three_d = True, Ready = True):
    
    """
    This fucntion generates a 2D and a 3D plot of tofd for any given C-Scan. 
    Note:This function takes the pickle file that was output from the "read_cscan" function as input.
    
    Go to the Damage Visualization Section to edit the Z-axis threshholds. Refer to plot_3D_scan function 
    for further guidance on this.
    
    """
    
    #Inputs include the pickle file that was output from the "read_cscan" function, a y=1 n=0 for which plots to generate
    #and location of the section cut (-1=program detect, 0=no plot, or <float> = scan location for section cut)
    
    # Sample function call copy/pastes:
    # cscan_plots("C:\\Users\\macpd\\Documents\\PhD\\C-Scan_Data\\astm_impact_specimens\\UTwin_output\\lvi-thp-08-009-CH1-plot.pk",tof1plot = 0, tofdplot = 0, amp1plot = 0, amp2plot = 0, amp3plot = 0, contplot = 1, sect_cut = 36.0, cont_range = [-0.8,0.8])
    # cscan_plots("C:\\Users\\macpd\\Documents\\PhD\\C-Scan_Data\\astm_impact_specimens\\UTwin_output\\lvi-thp-08-009-CH1-plot.pk",tof1plot = 1, tofdplot = 1, amp1plot = 1, amp2plot = 1, amp3plot = 1, contplot = 1, sect_cut = -1, cont_range = [-1.0,1.0])
    
    #start a timer for performance tracking
    tstart = time.time()
    date_str = str(datetime.now())
    print('Program initiated at ' + date_str + '\n')
    
    #Load in pickle file, an output from the function "read_cscan"
    pk_file = open(pkfile,'rb')
    pk_plot = pickle.load(pk_file)
    pk_file.close()
    #Re-categorize data from pickel file into separate variables
    tof1,tofd,amp1,amp2,amp3,p_index,p_scan,plies,index_res,scan_res,file = pk_plot
    #------USER INPUTS------------------------------------------------------------------------------------------------------------------
    #edge_1 and edge_2 help define how to "un-tilt" the surface profile. The program will take points that fall between edge_1 and 
    #edge_2 from the center line of each of the 4 sides of the scan, these points will then define a flat plane. The definition of that plane
    #is then used to apply a factor on the time of arrival for each data point, so that the plotted surface profile will be flat, and deformations
    #are not overwhelmed by the global slope of the panel.
    edge_1 = 50
    edge_2 = 100
    
    
    
    #---Process 1st echo---
    #adjust for panel tilt and calc dent depth
    #Tilt calculated by taking linear regression along two perpindicular lines
    #through the center of the panel and offsetting 1st peak by those curves

    scan_edge = tof1[int(len(tof1)/2.2)]
    index_edge = [tof1[ii][int(len(tof1[ii])/2)] for ii in range(len(tof1))]
    index_tilt = np.polyfit(p_index[edge_1:edge_2]+p_index[-edge_2:-edge_1],index_edge[edge_1:edge_2]+index_edge[-edge_2:-edge_1],1) #Just taking the first and last 15 points from list to prioritize edges
    scan_tilt = np.polyfit(p_scan[edge_1:edge_2]+p_scan[-edge_2:-edge_1],scan_edge[edge_1:edge_2]+scan_edge[-edge_2:-edge_1],1)
    
    s_profile = [[0.0]*len(p_scan) for ii in range(len(p_index))]
    s_profile_list = []
    tof_tot = [] #Get all the tofd into one big list, to guess at a baseline thickness

    tofd_min = 10000.
    tofd_max = -100.
    for ii in range(len(tof1)):
        for jj in range(len(tof1[ii])):
            if tof1[ii][jj] != 0:
                #Un-tilt front surface measurements
                s_profile[ii][jj] = tof1[ii][jj] - index_tilt[0]*p_index[ii] - index_tilt[1] - scan_tilt[0]*p_scan[jj] - scan_tilt[1]
                #Add points to the baseline profile list, not too close to edge or center
                #Also compiling list of TOF2-TOF1, for baseline thickness
                if (ii > 0.125*len(tof1) and ii < 0.3*len(tof1)) or (ii > 0.7*len(tof1) and ii < 0.875*len(tof1)):
                    if (jj > 0.125*len(tof1[ii]) and jj < 0.3*len(tof1[ii])) or (jj > 0.7*len(tof1[ii]) and jj < 0.875*len(tof1[ii])):
                        s_profile_list += [s_profile[ii][jj]]
                        tof_tot += [tofd[ii][jj]]
                if tofd[ii][jj] > tofd_max:
                    tofd_max = tofd[ii][jj]
                if tofd[ii][jj] < tofd_min:
                    tofd_min = tofd[ii][jj]
    
    #Average TOF
    tofd_avg = np.mean(tofd)
    print("Avg TOFD: ", tofd_avg)
    print(plies)
    #Surface profile untilted pretty well by now, but we want the edges to be "sea level" so to speak, so shift everything
    #such that the edges are as close to zero depth as possible
    scan_edge = s_profile[int(len(s_profile)/2)]
    index_edge = [s_profile[ii][int(len(s_profile[ii])/2)] for ii in range(len(s_profile))]
    #s_baseline = np.mean(scan_edge[5:15]+scan_edge[-15:-5]+index_edge[5:15]+index_edge[-15:-5]) #Use for deformed (not dented) panel
    s_baseline = np.mean(scan_edge[edge_1:edge_2]+scan_edge[-edge_2:-edge_1])

    

    #Manually defining max and min of surface contour plot:
        #Comment these out if you want to use the actual max/min calculated in the loop above this
    surf_min = cont_range[0]
    surf_max = cont_range[1]
    # surf_max = -1.0*surf_min
       
    #---Calculate Damage Area---
    #Need to scan through a second time to check for damage. If a string of data points only has one peak, and it is bounded by peaks of "damage",
    #then assign those data points to be damaged areas as well.
    index_min = 0.1
    index_max = 0.8
    scan_min = 0.1
    scan_max = 0.9
    #Assuming a good estimate for full thickness tof = median of tof_tot
    tof_thk = np.median(tof_tot)
    print("Baseline tof (median/mean):",tof_thk,np.mean(tof_tot))
    #tlevels = np.linspace(0,tof_thk,plies) #make each color on contour plot correspond to ply change (this doesn't really work yet, so commented out for now)
    
    #Empty z-axis variable for damage / no damage map
    dmg_plot = [[0.5]*len(p_scan) for ii in range(len(p_index))]
    dmg_area = 0.0 #track total damage area
    unit_area = (index_res)*(scan_res) #How much physical area one data point represents (not accurate on edges of region but that should be irrelevant)
    for ii in range(0,len(tofd)):
        for jj in range(0,len(tofd[ii])):
            #Check to see if it is in the user defined area for damage
            if ii > index_min*len(tofd) and ii < index_max*len(tofd):
                if jj > scan_min*len(tofd[ii]) and jj < scan_max*len(tofd[ii]):
                    #Make sure there is some signal at point (not off panel)
                    if tofd[ii][jj] > 0.0:
                        #Threshold for damage - tof a ply thickness away from back face
                        if tofd[ii][jj] < ((plies-1.95)/(plies))*tof_thk:
                            dmg_plot[ii][jj] = 1.0
                else:
                    dmg_plot[ii][jj] = 0.0
            else:
                dmg_plot[ii][jj] = 0.0
                
    #Scan through a second time to see if damage is present just by an unreadable signal, bounded by readable levels of damage
    #Update- need to make it so that if it is within x points of a damaged point, then just make it damage too
    #Make it 5mm away, so 20 points at 0.25mm resolution. Probably still need to scan 2 times, once in each direction?
    prox_chk = 10
    for ii in range(0,len(tofd)):
        for jj in range(0,len(tofd[ii])):
            if tofd[ii][jj] < 0.0: #if the signal has no TOFD value, check if there is damage nearby, assign as damage if there is
                
                if ii > index_min*len(tofd) and ii < index_max*len(tofd): #Stay within user defined region
                    if jj > scan_min*len(tofd[ii]) and jj < scan_max*len(tofd[ii]): #Stay within user defined region
                        for aa in range(ii-prox_chk,ii+prox_chk+1):
                            for bb in range(jj-prox_chk,jj+prox_chk+1):
                                if dmg_plot[aa][bb] == 1.0:
                                    dmg_plot[ii][jj] = 1.0
    #Sum total damage area:
    for ii in range(0,len(dmg_plot)):
        for jj in range(0,len(dmg_plot[ii])):
            if dmg_plot[ii][jj] == 1.0:
                dmg_area += unit_area
            # if dmg_plot[ii][jj] == 1.0 and dmid == 0: #if a data point shows damage not preceded by other unbounded damage, place a marker
            #     dstart = jj
            # if tofd[ii][jj] == 0.0 and dstart > 0: #if data shows no 2nd echo after a damage, place a marker
            #     dmid = 1
            # if dmg_plot[ii][jj] == 1.0 and dmid == 1: #if data shows damage after some unbounded damage, change all those points to show as damage
            #     for kk in range(dstart,jj):
            #         if ii > index_min*len(tofd) and ii < index_max*len(tofd):
            #             if jj > scan_min*len(tofd[ii]) and jj < scan_max*len(tofd[ii]):
            #                 dmg_plot[ii][kk] = 1.0
            #                 dmg_area += unit_area
            #             else:
            #                 dmg_plot[ii][jj] = -1.0
            #         else:
            #             dmg_plot[ii][jj] = -2.0   
            #     dstart = 0
            #     dmid = 0
    #-#-#-#-#

    print('Damage area: ' + '{0:.4f}'.format(dmg_area))

    #Save damage info and dent profile data in txt file for later plotting
    g = open(pkfile[:-7] + 'out.txt', 'w')
    g.write('Processed C-scan File:\t' + file[len(file) - file[::-1].index('\\'):] + '\n')
    g.write('Damage Area:          \t' + '{0:.3f}'.format(dmg_area) + '\tmm\n')
    # g.write('Projected dent area:  \t' + '{0:.3f}'.format(area_count) + '\tmm^2\n')
    # g.write('Local deformed volume:\t' + '{0:.4f}'.format(volume_count) + '\tmm^3\n')
    # g.write('Max Dent Depth:       \t' + '{0:.3f}'.format(min(s3)) + '\tmm\n')
    g.write('Baseline TOF:         \t' + '{0:.3f}'.format(tof_thk) + '\tus\n')
    # g.write('Section cut taken at: \t' + '{0:.2f}'.format(scut_index) + '\tmm (index)\n')
    g.write('Position\tSmoothed Curve\tOriginal Curve\n')
    # for ii in range(len(s2)):
    #     g.write('{0:.3f}'.format(p_scan[ii]) + '\t' + '{0:.3f}'.format(s_profile[section_cut[0]][ii]) + '\t' + '{0:.3f}'.format(s3[ii]) + '\n')
    g.close()
    
    #Assuming a good estimate for full thickness tof = median of tof_tot
    print("Baseline tof (median/mean):",tof_thk,np.mean(tof_tot))

    #### TOF + Depth Plots Code ####
    
    #---Prepare TOFD data for plotting---
    #Convert TOF from us to thickness when requested (damage calc above still used raw TOF)
    print("Original TOF_max:", tofd_max)
    if thk != 'tof':
        tof2thk = []
        for ii in range(len(tofd)):
            tof2thk += [(tofd[ii]/tof_thk)*thk]
        tofd = tof2thk
        print("Original Thk_max:", np.max(tofd))

    #Recompute extremes in the same units as tofd (us or mm) before percentile filtering
    tofd_min = 10000.
    tofd_max = -100.
    for ii in range(len(tofd)):
        for jj in range(len(tofd[ii])):
            if tofd[ii][jj] > 0.0:
                if tofd[ii][jj] > tofd_max:
                    tofd_max = tofd[ii][jj]
                if tofd[ii][jj] < tofd_min:
                    tofd_min = tofd[ii][jj]

    if len(tofd) == 0:
        print("WARNING: tofd is empty")
    else:
        counts, bin_edges = np.histogram(tofd, bins=200)
        max_bin_index = np.argmax(counts)
        mode_estimate = 0.5 * (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1])

        # Current threshold set to 20% above mode
        if tofd_max > 1.2*mode_estimate:
            tofd_max = np.nanpercentile(tofd, 98)
            
            if thk != 'tof':
                print("98th Percentile Thk_max:", tofd_max)
            else:
                print("98th Percentile TOF_max:", tofd_max)
                

    #---Generate Plots---
    #Playing around with some plotting options here, so a lot of attempts are just commented out because I'm too committed to just delete them
    #
    #Test number:
    testid = pkfile[len(pkfile) - pkfile[::-1].index('\\'):-8]
    
    if tofdplot == 1:
        if thk == 'tof':
            levels = list(np.linspace(0,tofd_max,13))
            fig, ax = plt.subplots()
            plot1 = ax.contourf(p_scan,p_index,tofd, 10, levels = levels, cmap = 'jet')
            ax.set_title('TOF: 2nd Peak - 1st Peak')
            ax.set_ylabel('Index Axis (mm)')
            ax.set_xlabel('Scanning Axis (mm)')
            ax.set_aspect('equal')
            fig.colorbar(plot1,ax = ax, label = 'TOF (microseconds)', orientation = 'horizontal')
        else:
            levels = list(np.linspace(0,thk*1.05,25))
            fig, ax = plt.subplots()
            plot1 = ax.contourf(p_scan,p_index,tofd, 10, levels = levels, cmap = 'jet')
            ax.set_title('TOF: 2nd Peak - 1st Peak')
            ax.set_ylabel('Index Axis (mm)')
            ax.set_xlabel('Scanning Axis (mm)')
            ax.set_aspect('equal')
            thickness_units = '(mm)' #Can also change to "(plies)" and just have it report fractions of a ply
            fig.colorbar(plot1,ax = ax, label = 'Thickness ' + thickness_units, orientation = 'horizontal')

        # fig.canvas.set_window_title(testid + '-tofd')
        fig.canvas.manager.set_window_title(testid + '-tofd')
        plt.savefig(pkfile[:-7] + 'tofd.png')
        
    if three_d:
        
        """# # # = = = Damage Visualization Section = = = # # #"""
        
        ### User Inputs: Data Selection Coefficients ###
        
        max_coeff = 0.87
        min_coeff = 0.97
        radius = 60
        
        # Top View
        plot_3D_cscan(None,None,None,
                     '3D C-scan (Top View)',
                     '3D_Top.png',pkfile, 90,-90, thk,
                     max_coeff, min_coeff, radius,
                     p_scan, p_index, tofd, tofd_max, tofd_min,
                     Front,None)
        
        if Ready:       
            
            # Isometric View
            plot_3D_cscan(None,None,None,
                         '3D C-scan',
                         '3D.png',pkfile, None, None, thk,
                         max_coeff, min_coeff, radius,
                         p_scan, p_index, tofd, tofd_max, tofd_min,
                         Front,True)
            
            # Front Side View
            plot_3D_cscan(None,None,None,
                         '3D C-scan (Front-Side View)', 
                         '3D_Side.png', pkfile, 0, -90, thk, 
                         max_coeff, min_coeff, radius,
                         p_scan, p_index, tofd, tofd_max, tofd_min,
                         Front,None)
            
            #Save 3D Data Processing Threshold in txt file for reference
            g = open(pkfile[:-7] + '3D-thresholds.txt', 'w')
            g.write('Processed C-scan File:\t' + file[len(file) - file[::-1].index('\\'):] + '\n')
            if thk == 'tof':
                g.write(f'Plot is in terms of:    \t{thk}\n')
                g.write('Maximum TOFd:            \t' + '{0:.3f}'.format(tofd_max) + '\t[us]\n')
                g.write('Upper TOFd limit:        \t' + '{0:.3f}'.format(max_coeff) + '\t[us]\n')
                g.write('Lower TOFd limit:        \t' + '{0:.3f}'.format(min_coeff) + '\t[us]\n')
                g.write('Damage Radius:           \t' + '{0:.3f}'.format(radius) + '\tpixels\n')
    
            else:
                g.write('Plot is in terms of thickness:    \t' + '{0:.3f}'.format(thk) + '\t[mm]\n')
                g.write('Maximum Thickness:      \t' + '{0:.3f}'.format(tofd_max) + '\t[mm]\n')
                g.write('Upper Thickness limit:  \t' + '{0:.3f}'.format(max_coeff) + '\t[mm]\n')
                g.write('Lower Thickness limit:  \t' + '{0:.3f}'.format(min_coeff) + '\t[mm]\n')
                g.write('Damage Radius:          \t' + '{0:.3f}'.format(radius) + '\tpixels\n')
    
            g.close()

    
    print('(time elapsed: ' + '{0:.4f}'.format(time.time()-tstart) + 's)\n')
        
def hybrid_cscan_plots(pkfile_front,pkfile_back,thk = 'tof',dx_manual=-0.2,dy_manual=0.2,dz_manual = 0.09,
                       auto_align=True,Manual_Auto = False, Ready = True):  
    
    """
    Combines the scan, axis, and tofd coordinates (Xf,Yf,Zf) of any set of Front and Back C-Scans
    into a single hybrid 3D plot. 
    Note: this function takes the hybrid_pickle files output from the plot_3D_scan funtion. That means you need to 1st process
    each Front and Back data individually and then define pkfile_front and pkfile_back as their respective pickle file paths
    
    This function performs automatic and manual alignments of the Front and Back scans:
        
        - Set auto_align to True if you want automatic alignment
        - Set manual_auto to True if you want to still use automatic alignment and add manual adjustments on top of that
        - Set both auto_align and manual_auto to False if you only wish to align the scans manually (not recommended)
        
    Finally, set Ready to True if you are satisfied with your data alignment. This will produce 
    Multiple final views of the hybrid 3D C-Scan and an additional Desity plot of the percolation pathways within the specimen
        
    """
    
    #start a timer for performance tracking
    tstart = time.time()
    date_str = str(datetime.now())
    print('Program initiated at ' + date_str + '\n')
    
    #Load in pickle file, an output from the function "read_cscan"
    front = open(pkfile_front,'rb')
    pk_plot_front= pickle.load(front)
    front.close()
    #Re-categorize data from pickel file into separate variables
    Coordinates_front = pk_plot_front
    edge_1 = 50
    edge_2 = 100
    
    #Load in pickle file, an output from the function "read_cscan"
    back = open(pkfile_back,'rb')
    pk_plot_back = pickle.load(back)
    back.close()
    #Re-categorize data from pickel file into separate variables
    Coordinates_back = pk_plot_back
    edge_1 = 50
    edge_2 = 100
    
    # Mirror Back Surface C Scan about y axis
    x_center = np.mean(pk_plot_back["Xf"])
    X_back_flipped = 2*x_center - pk_plot_back["Xf"]
    
    Y_back_flipped = pk_plot_back["Yf"]
    Z_back_flipped = pk_plot_back["Zf"]
    front_centroid = pk_plot_front["centroid_xyz"]
    back_centroid  = pk_plot_back["centroid_xyz"]
    
    # # # = = = Centroid & Midplane Alignment Section Begins = = = # # #
    
    dx = front_centroid[0] - back_centroid[0]
    dy = front_centroid[1] - back_centroid[1]
    print(f"front_midplane = {front_centroid[2]: .3f}, back_midplane = {back_centroid[2]: .3f}")
    
    # if thk == 'tof':
    #     dz = 0.07 - (front_centroid[2] - back_centroid[2])
    # else:
    #     dz = 0.2 - (front_centroid[2] - back_centroid[2])
    
    if thk == 'tof':
        dz = 0.07
    else:
        dz = 0.2
    
    if auto_align:
        X_back_aligned = X_back_flipped + dx
        Y_back_aligned = Y_back_flipped + dy
        Z_back_aligned = Z_back_flipped + dz
        print("Automatic centroid shift:")
        print(f"dx_auto = {dx:.3f}, dy_auto = {dy:.3f}, dz_auto = {dz:.3f}")

        
        # Enable this if you want to add a manual shift on top of the automatic shift
        if Manual_Auto:
            
            X_back_aligned += dx_manual
            Y_back_aligned += dy_manual
            Z_back_aligned += dz_manual  
            print("Automatic + Manual applied shift:")
            print(f"dx_total = {dx + dx_manual:.3f}, dy_total = {dy + dy_manual:.3f}, dz_total = {dz + dz_manual:.3f}")
    
    else:
        X_back_aligned = X_back_flipped
        Y_back_aligned = Y_back_flipped
        Z_back_aligned = Z_back_flipped
        
        X_back_aligned += dx_manual
        Y_back_aligned += dy_manual
        Z_back_aligned += dz_manual
        
        print("Manual user shift:")
        print(f"dx_manual = {dx_manual:.3f}, dy_manual = {dy_manual:.3f}, dz_manual = {dz_manual:.3f}")
    
    ### === Outline Alignment Zone Begins === ###
    
    # Separate front and back aligned data
    X_front = pk_plot_front["Xf"]
    Y_front = pk_plot_front["Yf"]
    
    X_back  = X_back_aligned
    Y_back  = Y_back_aligned
    
    # Compute outlines
    cont_front, xi_f, yi_f = compute_damage_outline(X_front, Y_front)
    cont_back,  xi_b, yi_b = compute_damage_outline(X_back, Y_back)
    
    plt.figure(figsize=(7,7))

    # Plot front outline
    for i, contour in enumerate(cont_front):
        plt.plot(
            xi_f[contour[:, 1].astype(int)],
            yi_f[contour[:, 0].astype(int)],
            color='blue',
            linewidth=2,
            label='Front' if i == 0 else ""
        )
    
    # Plot back outline
    for i, contour in enumerate(cont_back):
        plt.plot(
            xi_b[contour[:, 1].astype(int)],
            yi_b[contour[:, 0].astype(int)],
            color='red',
            linewidth=2,
            label='Back' if i == 0 else ""
        )
    
    plt.xlabel("Scanning Axis (mm)")
    plt.ylabel("Index Axis (mm)")
    plt.title("Damage Outline Comparison (Top View)")
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # # === Outline Alignment Zone Ends === # #
    
    # # = = = Z Alignment Check = = = # # 
    
    # Concatenate Front and Back Coordinates
        
    X_combined = np.concatenate([pk_plot_front["Xf"], X_back_aligned])
    Y_combined = np.concatenate([pk_plot_front["Yf"], Y_back_aligned])
    Z_combined = np.concatenate([pk_plot_front["Zf"], Z_back_aligned])
    
    print("Saving Hybrid 3D Plotting Combined Data")    
    Coordinates = {
        "X": X_combined,
        "Y": Y_combined,
        "Z": Z_combined,
    }
    outfile = pkfile_front.replace(".pk", "_plotting_data.pk")
    with open(outfile, "wb") as f:
        pickle.dump(Coordinates, f)
    
    ### === Final Front and Back Scan Side View Comparison === ###
        
    # Front
    plot_3D_cscan(pk_plot_front["Xf"],pk_plot_front["Yf"],pk_plot_front["Zf"],
                    'Final Front 3D C-scan (Front-Side View)',
                    'Final_Front_3D_Front-Side.png',pkfile_front, 0, -90, thk,
                    None, None, None,None, None,
                    None,None,
                    None,None)
        
    # Back
    plot_3D_cscan(X_back_aligned,Y_back_aligned,Z_back_aligned,
                 'Final Back 3D C-Scan (Front-Side View)', 
                 'Final_Back_3D_Side.png',pkfile_front, 0, -90, thk, 
                 None, None, None,None, None,
                 None,None,
                 None,None)
    
    # # = = = Z Alignment Check Ends = = = # # 
    
    # # # = = = Centroid & Midplane Alignment Section Ends = = = # # #
    
    if Ready:
    
        # # # ===== Plotting Section Begins ==== # # # 
        
        fig3d = plt.figure(figsize=(9,7))
        ax3d = fig3d.add_subplot(111, projection='3d')

        sc = ax3d.scatter(
            X_combined,
            Y_combined,
            Z_combined,
            c=Z_combined,
            cmap='jet',
            s=0.3,
            depthshade=False
        )
        
        ax3d.set_title('Hybrid 3D C-Scan')
        ax3d.set_xlabel('Scanning Axis (mm)')
        ax3d.set_ylabel('Index Axis (mm)')
        
        if thk == 'tof':
            ax3d.set_zlabel('TOF Difference (µs)')
        else:
            ax3d.set_zlabel('Thickness (mm)')
        
        testid = pkfile_front[len(pkfile_front) - pkfile_front[::-1].index('\\'):-8]
        fig3d.canvas.manager.set_window_title(testid + '-hybrid-3D')
        plt.tight_layout()
        plt.savefig(pkfile_front[:-7] + 'Hybrid_3D.png')
        
         
        ### OTHER HELPFUL VIEWS ###
        
        ### === Hybrid Helper Views === ###
        
        # Comment as needed...
        
        # Top View
        plot_3D_cscan(X_combined,Y_combined,Z_combined,
                     'Hydrid 3D C-scan (Top View)',
                     'Hybrid_3D_Top.png',pkfile_front, 90, -90, thk,
                     None,None,
                     None, None, None,None, None,
                     None,None)
        
        # Bottom View
        plot_3D_cscan(X_combined,Y_combined,Z_combined,
                     'Hybrid 3D C-scan (Bottom View)',
                     'Hybrid_3D_Bottom.png',pkfile_front, -90,90, thk,
                     None,None,
                     None, None, None,None, None,
                     None,None)
        
        # Front Side View
        plot_3D_cscan(X_combined,Y_combined,Z_combined,
                     'Hybrid_3D C-scan (Front-Side View)', 
                     'Hybrid_3D_Side.png', pkfile_front, 0, -90, thk, 
                     None, None, None,None, None,
                     None,None,
                     None,None)
        
        # # # ===== Plotting Section Ends ==== # # # 
        
        # # # ==== Save rest of Hybrid 3D Outputs in txt file for reference === # # #
        g = open(pkfile_front[:-7] + 'out.txt', 'w')
        front_name = os.path.basename(pkfile_front)
        back_name  = os.path.basename(pkfile_back)
        g.write('Merged C-scan Files:\t' + front_name + ' | ' + back_name + '\n')
    
        if thk == 'tof':
            g.write(f'Plots are in terms of:    \t{thk}\n')
            x_c, y_c, z_c = front_centroid
            g.write(
                f'Centroid of Front C-Scan (x,y,z):\t'
                f'({x_c:.3f}, {y_c:.3f}, {z_c:.3f})\t[mm, mm, us]\n'
            )
            x_c, y_c, z_c = back_centroid
            g.write(
                f'Centroid of Back C-Scan (x,y,z):\t'
                f'({x_c:.3f}, {y_c:.3f}, {z_c:.3f})\t[mm, mm, us]\n'
            )
            g.write(f'Back Scan Alignment:\n')
            
            if auto_align:
                g.write(f'Auto Shift in the x-axis:   \t' + '{0:.3f}'.format(dx) + '\t[mm]\n')
                g.write(f'Auto Shift in the y-axis:   \t' + '{0:.3f}'.format(dy) + '\t[mm]\n')
                g.write(f'Auto Shift in the z-axis:   \t' + '{0:.3f}'.format(dz) + '\t[us]\n')
            else:
                g.write(f'Manual Shift in the x-axis:   \t' + '{0:.3f}'.format(dx_manual) + '\t[mm]\n')
                g.write(f'Manual Shift in the y-axis:   \t' + '{0:.3f}'.format(dy_manual) + '\t[mm]\n')
                g.write(f'Manual Shift in the z-axis:   \t' + '{0:.3f}'.format(dz_manual) + '\t[us]\n')

            if Manual_Auto:
                g.write(f'Manual Shift in the x-axis:   \t' + '{0:.3f}'.format(dx_manual) + '\t[mm]\n')
                g.write(f'Manual Shift in the y-axis:   \t' + '{0:.3f}'.format(dy_manual) + '\t[mm]\n')
                g.write(f'Manual Shift in the z-axis:   \t' + '{0:.3f}'.format(dz_manual) + '\t[us]\n')
            
        else:
            g.write('Plots are in terms of thickness:    \t' + '{0:.3f}'.format(thk) + '\t[mm]\n')
            x_c, y_c, z_c = front_centroid
            g.write(
                f'Centroid of Front C-Scan (x,y,z):\t'
                f'({x_c:.3f}, {y_c:.3f}, {z_c:.3f})\t[mm, mm, mm]\n'
            )
            x_c, y_c, z_c = back_centroid
            g.write(
                f'Centroid of Back C-Scan (x,y,z):\t'
                f'({x_c:.3f}, {y_c:.3f}, {z_c:.3f})\t[mm, mm, mm]\n'
            )
            g.write(f'Back Scan Alignment:\n')
            
            if auto_align:
                g.write(f'Auto Shift in the x-axis:   \t' + '{0:.3f}'.format(dx) + '\t[mm]\n')
                g.write(f'Auto Shift in the y-axis:   \t' + '{0:.3f}'.format(dy) + '\t[mm]\n')
                g.write(f'Auto Shift in the z-axis:   \t' + '{0:.3f}'.format(dz) + '\t[mm]\n')
            else:
                g.write(f'Manual Shift in the x-axis:   \t' + '{0:.3f}'.format(dx_manual) + '\t[mm]\n')
                g.write(f'Manual Shift in the y-axis:   \t' + '{0:.3f}'.format(dy_manual) + '\t[mm]\n')
                g.write(f'Manual Shift in the z-axis:   \t' + '{0:.3f}'.format(dz_manual) + '\t[mm]\n')

            if Manual_Auto:
                g.write(f'Manual Shift in the x-axis:   \t' + '{0:.3f}'.format(dx_manual) + '\t[mm]\n')
                g.write(f'Manual Shift in the y-axis:   \t' + '{0:.3f}'.format(dy_manual) + '\t[mm]\n')
                g.write(f'Manual Shift in the z-axis:   \t' + '{0:.3f}'.format(dz_manual) + '\t[mm]\n')


def density_plots(pkfile, Translate_Coor = True):
    
    """
    This function uses the X, Y, Z values for any 3D hybrid dataset 
    outputed from the hybrid_cscan_plot function as a pk file
    to generate a set of density plots.
    
        - This function uses another helper function called compute_percolation_grid,
        which takes the coordinates X, Y, Z, a user defined grid resolution (grid_res), 
        z tolerance (z_tol), and a special command called raw_data. 
        
            - by setting raw_data to True, the density plot counts the amount 
            of data points per grid cell, if raw_data is set to False or simply
            not defined, the density plot counts the amount of delamination 
            per grid cell. 
        
        - By setting Translate_Coor to True, the user can use a referance
        marker (origin) to translate the coordinates with respect to such marker   
        
    """
        
    # # # ===== Data Import Zone Begins ==== # # # 
        
    #Load in pickle file, an output from the function "hybrid_cscan_plot"
    hybrid_data = open(pkfile,'rb')
    pk_plot= pickle.load(hybrid_data)
    hybrid_data.close()
    #Re-categorize data from pickel file into separate variables

    
    # Define X, Y, Z coordinates
    X_combined = pk_plot["X"]   
    Y_combined = pk_plot["Y"]
    Z_combined = pk_plot["Z"]
    
    # # # ===== Data Import Zone Ends ==== # # #
    
    # # # ===== Plotting Zone Begins ==== # # #
    
    # Delamination Count
    layer_grid, x_edges, y_edges = compute_delamination_grid(
        X_combined,
        Y_combined,
        Z_combined,
        grid_res=0.75,
    )

    plt.figure(figsize=(7,6))
    plt.imshow(
        layer_grid,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        origin='lower',
        aspect='equal'
    )
    plt.colorbar(label='Distinct Z Layers')
    plt.title("Through-Thickness Delamination Density Map")
    plt.xlabel("Scanning Axis (mm)")
    plt.ylabel("Index Axis (mm)")
    plt.tight_layout()
    plt.show()
    plt.savefig(pkfile[:-7] + 'Delamination_Density_Map.png')
    
    if Translate_Coor:
        
        xo = 12.89;  # [mm]
        yo = 9.38;  # [mm]
        X_trans = X_combined - xo;
        Y_trans = Y_combined - yo;
        
        # Translated Percolation Paths
        layer_grid, x_edges, y_edges = compute_delamination_grid(
            X_trans,
            Y_trans,
            Z_combined,
            grid_res=0.75,
        )
        
        plt.figure(figsize=(7,6))
        plt.imshow(
            layer_grid,
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            origin='lower',
            aspect='equal'
        )
        plt.colorbar(label='No. of Data Points per Cell')
        plt.title("Through-Thickness Delamination Density Map")
        plt.xlabel("Scanning Axis (mm)")
        plt.ylabel("Index Axis (mm)")
        plt.tight_layout()
        plt.show()
        plt.savefig(pkfile[:-7] + 'Translated_Density_Map.png')
        
        
    # # === Percolation Paths Section Ends === #
    
    
def bscan_plots(pkfile, thk = 3.292, x_range = None, y_range = [44,46]):
    
    """
    
    This function allows to plot only a specified range of the X and Y axes 
    In other words, this function provides slice views from the whole hybrid 
    dataset contained in the pk file.
    
    Note: The pk file is obtained from the hybrid_cscan_plots function 
        
    """
        
    # # # ===== Data Import Zone Begins ==== # # # 
        
    #Load in pickle file, an output from the function "hybrid_cscan_plot"
    hybrid_data = open(pkfile,'rb')
    pk_plot= pickle.load(hybrid_data)
    hybrid_data.close()
    #Re-categorize data from pickel file into separate variables

    # Define X, Y, Z coordinates
    X_combined = pk_plot["X"]   
    Y_combined = pk_plot["Y"]
    Z_combined = pk_plot["Z"]
    
    # # # ===== Data Import Zone Ends ==== # # #
    
    # # # ===== Plotting Section Begins ==== # # # 
    
    # Keep all points by default
    mask = np.ones(len(X_combined), dtype=bool)
    
    # Apply X limits if requested
    if x_range is not None:
        xmin, xmax = x_range
        mask &= (X_combined >= xmin) & (X_combined <= xmax)
    
    # Apply Y limits if requested
    if y_range is not None:
        ymin, ymax = y_range
        mask &= (Y_combined >= ymin) & (Y_combined <= ymax)
    
    # Trim all coordinates consistently
    X_plot = X_combined[mask]
    Y_plot = Y_combined[mask]
    Z_plot = Z_combined[mask]

    # Orthographic 2D projections (no 3D perspective distortion)
    plot_2D_cscan_projection(
        X_plot, Y_plot, Z_plot,
        'top',
        'Hybrid C-Scan (Top View)',
        'Hybrid_Slice.png',
        pkfile,
        thk,
    )

    plot_2D_cscan_projection(
        X_plot, Y_plot, Z_plot,
        'front',
        'Hybrid Cross-Section Front View',
        'Hybrid_CS_Front.png',
        pkfile,
        thk,
    )

    plot_2D_cscan_projection(
        X_plot, Y_plot, Z_plot,
        'right',
        'Hybrid Cross-Section Right Side View',
        'Hybrid_CS_RS.png',
        pkfile,
        thk,
    )
    
    # # # ===== Plotting Section Ends ==== # # # 
    
    
    
###### ======= Helper Funtions ======= ######

def plot_2D_cscan_projection(X, Y, Z, view, title, filename, pkfile, thk, point_size=10):
    """
    Flat orthographic projection of hybrid C-scan points onto a 2D plane.

    view:
        'top'   -> X vs Y, color = Z  (plan view)
        'front' -> Y vs Z, color = Z  (look along scanning axis)
        'right' -> X vs Z, color = Z  (look along index axis)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if view == 'top':
        horiz, vert = X, Y
        xlabel = 'Scanning Axis (mm)'
        ylabel = 'Index Axis (mm)'
        equal_aspect = True
    elif view == 'front':
        horiz, vert = X, Z
        xlabel = 'Index Axis (mm)'
        ylabel = 'Thickness (mm)' if thk != 'tof' else 'TOF Difference (µs)'
        equal_aspect = False
    elif view == 'right':
        horiz, vert = Y, Z
        xlabel = 'Scanning Axis (mm)'
        ylabel = 'Thickness (mm)' if thk != 'tof' else 'TOF Difference (µs)'
        equal_aspect = False
    else:
        raise ValueError("view must be 'top', 'front', or 'right'")

    if thk == 'tof':
        cbar_label = 'TOF Difference (µs)'
    else:
        cbar_label = 'Thickness (mm)'

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        horiz,
        vert,
        c=Z,
        cmap='jet',
        s=point_size,
        linewidths=0,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if equal_aspect:
        ax.set_aspect('equal')

    fig.colorbar(sc, ax=ax, label=cbar_label, orientation='horizontal')
    testid = pkfile[len(pkfile) - pkfile[::-1].index('\\'):-8]
    fig.canvas.manager.set_window_title(testid + '-' + filename[:-4])
    plt.tight_layout()
    plt.savefig(pkfile[:-7] + filename)

def plot_3D_cscan(Xf,Yf,Zf,title, filename, pkfile, elev, azim, thk, max_coeff, min_coeff, radius, p_scan, p_index, tofd, tofd_max, tofd_min, Front, save_processed = False):
    
    """
    Generates 3D plots of x-axis vs. y-axis vs. tofd
    
    By seting Front = True, the function flips the Z axis
    to better match the damage position with respect to its real-life position.
    
    There are two different sets of inputs to generate a 3D plot: 
        
        1) Define Xf, Yf, Zf and set p_scan, p_index, tofd, tofd_max, tofd_min to None 
        to skip the Z Data filtering zone and simply plot Xf, Yf, Zf. Note: This is only
        recommended if the x,y,z data has been already process (like for a hybrid scan)
        
        2) Difine p_scan, p_index, tofd, tofd_max, tofd_min and set Xf, Yf, Zf to None
        to filter the Z data (tofd) and find the centroid of the scan
        
            - In this mode, you must define max_coeff and min_coeff to change 
            how much tofd data near the max and min limits
            is being selected as damage
        
    Moreover, there are a couple more useful user inputs:
        
        - Define elev and azim to change the orientation of the 3D plot. This is independent of 
        the method you choose to generate the plots. Set inputs to None for Isometric View
        
        - Increase or decrease the radius [pixels] depending on where the damage is with respect
        to the center of the scan and its size as well
    
    """
    
    testid = pkfile[len(pkfile) - pkfile[::-1].index('\\'):-8]
    
    if all(v is not None for v in [p_scan, p_index, tofd, tofd_max, tofd_min]):
   
       # ===== Z Data Filtering Zone  ===== 
       
        X, Y = np.meshgrid(p_scan, p_index)
        Z = np.array(tofd, dtype=float)
        
        
        # User Inputs
        # Edit the threshold coefficients to control how much data is included in the damage zone
        # The higher the coefficient, the more top/bottom surface data will be recognized as damage
        # For Example: A 100% threshold (1*tofd_max) will include the entire bottom surface data (tofd_max)
        # Modify as needed until your 3D plot shows no artifacts
        
        upper_thresh = max_coeff * tofd_max
        lower_thresh = min_coeff * tofd_min
        
        mask_extremes = (Z >= upper_thresh) | (Z <= lower_thresh)
        Z[mask_extremes] = np.nan
        
        # Binary mask of remaining points
        binary_mask = ~np.isnan(Z)
        # Label connected regions
        labeled_array, num_features = label(binary_mask)
        
        # Panel center (in index coordinates)
        center_row = Z.shape[0] / 2
        center_col = Z.shape[1] / 2
        
        # Max allowed distance from center (in pixels)
        # Adjust this depending on your scan size
        max_radius = radius   # <-- tune this
        
        damage_mask = np.zeros_like(Z, dtype=bool)
        
        for region_label in range(1, num_features + 1):
        
            region_indices = np.where(labeled_array == region_label)
    
            if len(region_indices[0]) == 0:
                continue
        
            # Compute centroid
            row_centroid = np.mean(region_indices[0])
            col_centroid = np.mean(region_indices[1])
        
            # Distance from panel center
            dist = np.sqrt((row_centroid - center_row)**2 +
                           (col_centroid - center_col)**2)
        
            if dist <= max_radius:
                damage_mask[region_indices] = True
        
        # Crop region
        Z_clean = Z.copy()
        Z_clean[~damage_mask] = np.nan
        
        if Front:
            Z_clean = tofd_max - Z_clean
        
        # Flatten cropped region
        Xf = X.flatten()
        Yf = Y.flatten()
        Zf = Z_clean.flatten()
        
        # Remove NaNs
        mask = ~np.isnan(Zf)
        Xf, Yf, Zf = Xf[mask], Yf[mask], Zf[mask]
        
        # Calculate Centroid
        x_centroid = np.mean(Xf)
        y_centroid = np.mean(Yf)
        # z_centroid = np.median(tofd)
        z_centroid = tofd_max/2
        
        # ===== End of Z Data Filtering Zone  ===== 
        
        # ===== 3D Plotting Zone  ===== 
        
        if elev is not None and azim is not None:
        
            fig3d = plt.figure(figsize=(9,7))
            ax3d = fig3d.add_subplot(111, projection='3d')
            
            sc = ax3d.scatter(
                Xf,
                Yf,
                Zf,
                c=Zf,
                cmap='jet',
                s=0.3,          # point size (key for "layered" look)
                depthshade=False
            )
        
            ax3d.view_init(elev, azim)
            
            ax3d.set_title(title)
            if thk == 'tof':
                ax3d.set_zlabel('TOF Difference (µs)')
            else:
                ax3d.set_zlabel('Thickness (mm)')
                
            plt.tight_layout()
            plt.savefig(pkfile[:-7] + filename)
        
        else:
            
            fig3d = plt.figure(figsize=(9,7))
            ax3d = fig3d.add_subplot(111, projection='3d')
            
            sc = ax3d.scatter(
                Xf,
                Yf,
                Zf,
                c=Zf,
                cmap='jet',
                s=0.3,          # point size (key for "layered" look)
                depthshade=False
            )
            
            ax3d.set_title(title)
            ax3d.set_xlabel('Scanning Axis (mm)')
            ax3d.set_ylabel('Index Axis (mm)')
            
            if thk == 'tof':
                ax3d.set_zlabel('TOF Difference (µs)')
            else:
                ax3d.set_zlabel('Thickness (mm)')
            
            fig3d.canvas.manager.set_window_title(testid + '-tofd-3D')
            plt.tight_layout()
            plt.savefig(pkfile[:-7] + 'tofd_3D.png')
    
        if save_processed:
            print("Writing Hybrid 3D Plotting Data")    
            Coordinates = {
                "Xf": Xf,
                "Yf": Yf,
                "Zf": Zf,
                "centroid_xyz": (x_centroid, y_centroid, z_centroid),
                "tofd_max": tofd_max,
                "tofd_min": tofd_min,
                "thk": thk,
            }
            outfile = pkfile.replace(".pk", "_hybrid.pk")
            with open(outfile, "wb") as f:
                pickle.dump(Coordinates, f)
        
    else:
        
        if elev is not None and azim is not None:
        
            fig3d = plt.figure(figsize=(9,7))
            ax3d = fig3d.add_subplot(111, projection='3d')
            
            sc = ax3d.scatter(
                Xf,
                Yf,
                Zf,
                c=Zf,
                cmap='jet',
                s=0.3,          # point size (key for "layered" look)
                depthshade=False
            )
        
            ax3d.view_init(elev, azim)
            
            ax3d.set_title(title)
            if thk == 'tof':
                ax3d.set_zlabel('TOF Difference (µs)')
            else:
                ax3d.set_zlabel('Thickness (mm)')
                
            plt.tight_layout()
            plt.savefig(pkfile[:-7] + filename)
            
        
        else:
            
            fig3d = plt.figure(figsize=(9,7))
            ax3d = fig3d.add_subplot(111, projection='3d')
            
            sc = ax3d.scatter(
                Xf,
                Yf,
                Zf,
                c=Zf,
                cmap='jet',
                s=0.3,          # point size (key for "layered" look)
                depthshade=False
            )
            
            ax3d.set_title(title)
            ax3d.set_xlabel('Scanning Axis (mm)')
            ax3d.set_ylabel('Index Axis (mm)')
            if thk == 'tof':
                ax3d.set_zlabel('TOF Difference (µs)')
            else:
                ax3d.set_zlabel('Thickness (mm)')
            
            fig3d.canvas.manager.set_window_title(testid + '-tofd-3D')
            plt.tight_layout()
            plt.savefig(pkfile[:-7] + 'tofd_3D.png')
            
        x_centroid = np.mean(Xf)
        y_centroid = np.mean(Yf)
        z_centroid = np.mean(Zf)
            
    
        if save_processed:
            print("Writing Hybrid 3D Plotting Data")    
            Coordinates = {
                "Xf": Xf,
                "Yf": Yf,
                "Zf": Zf,
                "centroid_xyz": (x_centroid, y_centroid, z_centroid),
                "tofd_max": tofd_max,
                "tofd_min": tofd_min,
                "thk": thk,
            }
            outfile = pkfile.replace(".pk", "_hybrid.pk")
            with open(outfile, "wb") as f:
                pickle.dump(Coordinates, f)
                
def compute_damage_outline(X, Y, grid_res=0.5):
    
    """
    Converts scattered damage points into a 2D binary mask
    and extracts outline contours.
    """

    # Create grid
    xi = np.arange(np.min(X), np.max(X), grid_res)
    yi = np.arange(np.min(Y), np.max(Y), grid_res)
    Xg, Yg = np.meshgrid(xi, yi)

    # Binary occupancy grid
    mask = np.zeros_like(Xg, dtype=int)

    # Map each point to nearest grid index
    x_idx = np.searchsorted(xi, X) - 1
    y_idx = np.searchsorted(yi, Y) - 1

    valid = (
        (x_idx >= 0) & (x_idx < len(xi)) &
        (y_idx >= 0) & (y_idx < len(yi))
    )

    mask[y_idx[valid], x_idx[valid]] = 1

    # Slight dilation helps close small holes
    mask = binary_erosion(mask == 1) == False

    # Extract contours
    contours = find_contours(mask, 0.5)

    return contours, xi, yi
                
def compute_delamination_grid(
    X,
    Y,
    Z,
    grid_res=0.5,
    z_tol=0.07
):

    # Define grid
    x_edges = np.arange(
        np.floor(np.min(X)),
        np.ceil(np.max(X)) + grid_res,
        grid_res
    )

    y_edges = np.arange(
        np.floor(np.min(Y)),
        np.ceil(np.max(Y)) + grid_res,
        grid_res
    )

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    layer_grid = np.zeros((ny, nx), dtype=int)

    # Assign each point to a grid cell
    x_idx = np.clip(np.digitize(X, x_edges) - 1, 0, nx - 1)
    y_idx = np.clip(np.digitize(Y, y_edges) - 1, 0, ny - 1)

    for i in range(nx):
        for j in range(ny):

            mask = (x_idx == i) & (y_idx == j)

            Z_local = Z[mask]

            if len(Z_local) == 0:
                continue

            # Sort the local z-coordinates
            Z_sorted = np.sort(Z_local)

            # Count the number of distinct layers
            n_layers = 1

            for k in range(1, len(Z_sorted)):

                if Z_sorted[k] - Z_sorted[k - 1] > z_tol:

                    n_layers += 1

            layer_grid[j, i] = n_layers

    return layer_grid, x_edges, y_edges


# def count_layers(Z_local, z_tol):

#     if len(Z_local) == 0:
#         return 0

#     Z_sorted = np.sort(Z_local)

#     n_layers = 1

#     for k in range(1, len(Z_sorted)):

#         if Z_sorted[k] - Z_sorted[k - 1] > z_tol:

#             n_layers += 1

#     return n_layers

def endtune(): #to let me know when the code finishes in case I stop paying attention
    t = 500
    winsound.Beep(440*2, 1*t)
    winsound.Beep(466*2, 1*t)
    winsound.Beep(494*2, 1*t)
    winsound.Beep(523*2, 4*t)
