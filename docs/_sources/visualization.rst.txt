Visualization of obtained spectral functions
============================================

You may be able to use the plotting tools in 
https://github.com/yuzie007/ph_plotter/releases. 
Please download the latest one.
To use the tools, you need to run upho_sf with the option **--format hdf5**
and obtain **sf.hdf5**. ::

    /path/to/upho/scripts/upho_sf --fpitch 0.01 -s 0.05 --function lorentzian --format hdf5

Then run:: 

    DOWNLOADED_PATH/ph_plotter/scripts/band_sf.py --sf_with irreps --plot_style mesh --sf_max 0.8 --sf_min 0 --d_sf 0.2 --f_max 10 --f_min 0 --d_freq 2 --colormap_p r

Then the total spectral functions may be plotted as **band_sf_THz_w_bar.pdf**.

To plot decomposed spectral functions according to small representations, 
we first need to know **pointgroup_symbol** and **ir_labels**. 
For example, if you would like to focus on the point 2 on the band path 1,
The pointgroup_symbol of this point is found from the following command: ::

    % h5dump -d 1/2/pointgroup_symbol sf.hdf5
    HDF5 "sf.hdf5" {
    DATASET "1/2/pointgroup_symbol" {
       DATATYPE  H5T_STRING {
          STRSIZE 3;
          STRPAD H5T_STR_NULLPAD;
          CSET H5T_CSET_ASCII;
          CTYPE H5T_C_S1;
       }
       DATASPACE  SCALAR
       DATA {
       (0): "mm2"
       }
    }
    }

Then we find the pointgroup_symbol for this point is “mm2”.
The SRs for this point is found as, ::

    % h5dump -d 1/2/ir_labels sf.hdf5
    HDF5 "sf.hdf5" {
    DATASET "1/2/ir_labels" {
       DATATYPE  H5T_STRING {
          STRSIZE 2;
          STRPAD H5T_STR_NULLPAD;
          CSET H5T_CSET_ASCII;
          CTYPE H5T_C_S1;
       }
       DATASPACE  SIMPLE { ( 4 ) / ( 4 ) }
       DATA {
       (0): "A1", "A2", "B1", "B2"
       }
    }
    }

Then we find “A1”, “A2”, “B1”, and “B2” is the possible SRs for this point.
Note that some of them can have no contribution to the spectral function.

The decomposed spectral functions for B1 for the pointgroup_symbol of mm2 may be plotted by adding **—selected_irreps '{"mm2":["B1”]}’** for the plotting command as, ::

    DOWNLOADED_PATH/ph_plotter/scripts/band_sf.py --plot_style mesh --sf_max 0.8 --sf_min 0 --d_sf 0.2 --f_max 10 --f_min 0 --d_freq 2 --colormap_p r --selected_irreps '{"mm2":["B1"]}'
    
