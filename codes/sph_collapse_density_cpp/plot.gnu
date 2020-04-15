set xrange [-1:1]
set yrange [-1:1]
set zrange [-1:1]
set view equal xyz
set hidden3d

# plot a cut along the x axis where x < 0
spl 'density.txt' u ($1<0?$1:1/0):2:3:4 every 1 w p pt 7 ps 1 lc palette z
