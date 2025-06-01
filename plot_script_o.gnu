set title 'Radar Tracking Visualization'
set xlabel 'X (meters)'
set ylabel 'Y (meters)'
set grid
set xrange [-1000:15000]
set yrange [-1000:15000]
set key outside
set mouse mouseformat 3
set mouse zoomcoordinates
set mouse zoomfactors 1.1, 1.1
set title 'Radar Tracking Visualization\nLeft-click+drag: zoom, Middle-click: restore view, Right-click: context menu'
set terminal wxt enhanced size 1200,800 font 'Arial,10' persist raise
bind "r" "set xrange [-1000:15000]; set yrange [-1000:15000]; replot"
plot \
'plot_data_o.txt' using 1:2:(strcol(3) eq "T" ? 1 : 0) with lines lc rgb 'blue' title 'Tracks', \
'plot_data_o.txt' using 1:2:(strcol(3) eq "X" ? 1 : 0) with points pt 4 ps 1.5 lc rgb 'green' title 'True States', \
'plot_data_o.txt' using 1:2:(strcol(3) eq "M" ? 1 : 0) with points pt 7 ps 1.0 lc rgb 'red' title 'Measurements'
print "\n=== INTERACTIVE CONTROLS ===\n"
print "- Left-click and drag: zoom into selected region"
print "- Middle-click: reset to original view"
print "- Right-click: open context menu with more options"
print "- Press 'r' key: reset zoom to original range"
print "- Mouse wheel: zoom in/out"
print "==============================\n"
