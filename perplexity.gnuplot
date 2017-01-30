set title "Cross perplexity of test data for the first 10 epochs"
# png transparent enhanced
set terminal png enhanced giant font "arial,18" size 1600, 1200
set output "perplexity.png"

# set datafile separator "\t"

#set xtics nomirror rotate by -45
set xrange [1:10]
#set yrange [0:]
set xlabel "Epoch"
set ylabel "Perplexity"

plot 'perplexity.tsv' using 1:(2**$2) title 'Cross perplexity (CS)' with linespoints lw 2, \
     ''               using 1:(2**$3) title 'Cross perplexity (EN)' with linespoints lw 2
