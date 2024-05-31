# Electrodes lists
# -----------------
all_electrodes = [
    "LPA", "RPA", "Nz", "Fp1", "Fpz", "Fp2", "AF9", "AF7", "AF5", "AF3", "AF1", "AFz", 
    "AF2", "AF4", "AF6", "AF8", "AF10", "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", 
    "F6", "F8", "F10", "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", 
    "FT8", "FT10", "T9", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10", 
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P9", 
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10", "PO9", "PO7", "PO5", 
    "PO3", "PO1", "POz", "PO2", "PO4", "PO6", "PO8", "PO10", "O1", "Oz", "O2", "O9", 
    "Iz", "O10", "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2"
]

electrodes_experiments = ['Fp1','Fpz','Fp2','AF7','AF3','AF4','AF8',
                          'F9','F7','F5','F3','F1','Fz','F2','F4','F6','F8','F10',
                          'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8',
                          'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
                          'TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8',
                          'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
                          'PO7','PO3','POz','PO4','PO8',
                          'O1','Oz','O2']

# Electrodes clusters
# -------------------
# Electrodes by region
regions = {
    "frontal": ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F9', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'F10'],
    "temporal": ['FT7', 'T7', 'TP7', 'FT8', 'T8', 'TP8'],
    "central": ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],
    "parietal": ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
    "occipital": ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
}

# Electrodes by hemisphere
hemisphere = {
    "left": [e for e in electrodes_experiments if any(e.endswith(str(n)) for n in ['1', '3', '5', '7', '9'])],
    "midline": [e for e in electrodes_experiments if 'z' in e],
    "right": [e for e in electrodes_experiments if any(e.endswith(str(n)) for n in ['2', '4', '6', '8', '10'])]
}

# Combined list: (Electrode, Region, Hemisphere)
combined_list = []
for electrode in electrodes_experiments:
    region_found = [region for region, elist in regions.items() if electrode in elist]
    hemisphere_found = [hemi for hemi, elist in hemisphere.items() if electrode in elist]
    if region_found and hemisphere_found:
        combined_list.append((electrode, region_found[0], hemisphere_found[0]))

# Creating lists
electrodes_regions_positions = [region for electrode in electrodes_experiments for region, elist in regions.items() if electrode in elist]
electrodes_hemisphere_positions = [hemi for electrode in electrodes_experiments for hemi, elist in hemisphere.items() if electrode in elist]

# Map of indices for region-based and hemisphere-based sorting
# Sorting electrodes by regions then by hemisphere
sorted_by_region_then_hemi = sorted(combined_list, key=lambda x: (x[1], {"left": 1, "midline": 2, "right": 3}[x[2]]))
sorted_electrodes_by_region = [elec[0] for elec in sorted_by_region_then_hemi]

# Sorting electrodes by hemisphere
sorted_by_hemisphere = sorted(combined_list, key=lambda x: {"left": 1, "midline": 2, "right": 3}[x[2]])
sorted_electrodes_by_hemisphere = [elec[0] for elec in sorted_by_hemisphere]

# Generate index maps
region_indices = [electrodes_experiments.index(elec) for elec in sorted_electrodes_by_region]
hemisphere_indices = [electrodes_experiments.index(elec) for elec in sorted_electrodes_by_hemisphere]


# Electrodes positions and their shift
#-------------------------------------
shift_x = 10
shift_y = 10
shift_z = 0

all_electrodes_position = [
    (-86.0761, -19.9897, -47.9860), # LPA
    (85.7939, -20.0093, -48.0310),  # RPA
    (0.0083, 86.8110, -39.9830),    # Nz
    (-29.4367, 83.9171+shift_y, -6.9900),   # Fp1
    (0.1123, 88.2470+shift_y, -1.7130),     # Fpz
    (29.8723, 84.8959+shift_y, -7.0800),    # Fp2
    (-48.9708, 64.0872, -47.6830),  # AF9
    (-54.8397, 68.5722, -10.5900),  # AF7
    (-45.4307, 72.8622, 5.9780),    # AF5
    (-33.7007, 76.8371, 21.2270),   # AF3
    (-18.4717, 79.9041, 32.7520),   # AF1
    (0.2313, 80.7710, 35.4170),     # AFz
    (19.8203, 80.3019, 32.7640),    # AF2
    (35.7123, 77.7259, 21.9560),    # AF4
    (46.5843, 73.8078, 6.0340),     # AF6
    (55.7433, 69.6568, -10.7550),   # AF8
    (50.4352, 63.8698, -48.0050),   # AF10
    (-70.1019-shift_x, 41.6523, -49.9520),  # F9
    (-70.2629, 42.4743, -11.4200),  # F7
    (-64.4658, 48.0353, 16.9210),   # F5
    (-50.2438, 53.1112, 42.1920),   # F3
    (-27.4958, 56.9311, 60.3420),   # F1
    (0.3122, 58.5120, 66.4620),     # Fz
    (29.5142, 57.6019, 59.5400),    # F2
    (51.8362, 54.3048, 40.8140),    # F4
    (67.9142, 49.8297, 16.3670),    # F6
    (73.0431, 44.4217, -12.0000),   # F8
    (72.1141+shift_x, 42.0667, -50.4520),   # F10
    (-84.0759, 14.5673, -50.4290),  # FT9
    (-80.7750-shift_x, 14.1203, -11.1350),  # FT7
    (-77.2149, 18.6433, 24.4600),   # FC5
    (-60.1819, 22.7162, 55.5440),   # FC3
    (-34.0619, 26.0111, 79.9870),   # FC1
    (0.3761, 27.3900, 88.6680),     # FCz
    (34.7841, 26.4379, 78.8080),    # FC2
    (62.2931, 23.7228, 55.6300),    # FC4
    (79.5341, 19.9357, 24.4380),    # FC6
    (81.8151+shift_x, 15.4167, -11.3300),   # FT8
    (84.1131, 14.3647, -50.5380),   # FT10
    (-85.8941, -15.8287, -48.2830), # T9
    (-84.1611-shift_x, -16.0187, -9.3460),  # T7
    (-80.2801, -13.7597, 29.1600),  # C5
    (-65.3581, -11.6317, 64.3580),  # C3
    (-36.1580, -9.9839, 89.7520),   # C1
    (0.4009, -9.1670, 100.2440),    # Cz
    (37.6720, -9.6241, 88.4120),    # C2
    (67.1179, -10.9003, 63.5800),   # C4
    (83.4559, -12.7763, 29.2080),   # C6
    (85.0799+shift_x, -15.0203, -9.4900),   # T8
    (85.5599, -16.3613, -48.2710),  # T10
    (-85.6192, -46.5147, -45.7070), # TP9
    (-84.8302-shift_x, -46.0217, -7.0560),  # TP7
    (-79.5922, -46.5507, 30.9490),  # CP5
    (-63.5562, -47.0088, 65.6240),  # CP3
    (-35.5131, -47.2919, 91.3150),  # CP1
    (0.3858, -47.3180, 99.4320),    # CPz
    (38.3838, -47.0731, 90.6950),   # CP2
    (66.6118, -46.6372, 65.5800),   # CP4
    (83.3218, -46.1013, 31.2060),   # CP6
    (85.54889+shift_x, -45.5453, -7.1300),   # TP8
    (86.1618, -47.0353, -45.8690),  # TP10
    (-73.0093, -73.7657, -40.9980), # P9
    (-72.4343-shift_x, -73.4527, -2.4870),  # P7
    (-67.2723, -76.2907, 28.3820),  # P5
    (-53.0073, -78.7878, 55.9400),  # P3
    (-28.6203, -80.5249, 75.4360),  # P1
    (0.3247, -81.1150, 82.6150),    # Pz
    (31.9197, -80.4871, 76.7160),   # P2
    (55.6667, -78.5602, 56.5610),   # P4
    (67.8877, -75.9043, 28.0910),   # P6
    (73.0557+shift_x, -73.0683, -2.5400),   # P8
    (73.8947, -74.3903, -41.2200),  # P10
    (-54.9104, -98.0448, -35.4650), # PO9
    (-54.8404, -97.5279, 2.7920),   # PO7
    (-48.4244, -99.3408, 21.5990),  # PO5
    (-36.5114, -100.8529, 37.1670), # PO3
    (-18.9724, -101.7680, 46.5360), # PO1
    (0.2156, -102.1780, 50.6080),   # POz
    (19.8776, -101.7930, 46.3930),  # PO2
    (36.7816, -100.8491, 36.3970),  # PO4
    (49.8196, -99.4461, 21.7270),   # PO6
    (55.6666, -97.6251, 2.7300),    # PO8
    (55.7366, -98.1421, -35.5410),  # PO10
    (-29.4134, -112.4490-shift_y, 8.8390),  # O1
    (0.1076, -114.8920-shift_y, 14.6570),   # Oz
    (29.8426, -112.1560-shift_y, 8.8000),   # O2
    (-29.8184, -114.5700, -29.2160),# O9
    (0.0045, -118.5650, -23.0780),  # Iz
    (29.7416, -114.2600, -29.2560), # O10
    (-86.0761, -44.9897, -67.9860), # T3
    (85.7939, -45.0093, -68.0310),  # T5
    (-86.0761, -24.9897, -67.9860), # T4
    (85.7939, -25.0093, -68.0310)   # T6
]

electrodes_position_experiments = [all_electrodes_position[all_electrodes.index(elec)] for elec in electrodes_experiments]
