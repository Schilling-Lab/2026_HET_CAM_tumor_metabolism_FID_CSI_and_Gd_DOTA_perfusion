from hypermri import Study

# load in the complete experiment directory using the Study class
# Note: in windows you might have to use r"..." when providing the path
#
# 'scans' is now a dictionary-like object: each scan folder name (integer values
# by default) acts as a key  to the scan it contains, thee scan itself is read
# in as a BrukerExp object.
scans = Study(
    "/mnt/d/Data/20230217/2023.02.17 - sv - WEB1-00897 - HP Pyr PHIP DNP dyn csi/"
)

# to get an overview of all the scans loaded in, simply print it
print(scans)

# as an example, print every scans raw fid shape and the corresponding
# reconstructed image array shape.
# Since 'scans' is a dictionary we have to use .values() to iterate over all the
# BrukerExp objects.
for s in scans.values():
    print(f"{str(s.fid.shape):<10} --> {s.seq2d.shape}")
