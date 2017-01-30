.PHONY: clean

# Set your model names here
MODEL_CS=vybli-cs-features2-fixed-32-dense512+drop0.200000+dense512+drop0.200000+lstm512+drop0.200000-02-1.6890.h5
MODEL_EN=vybli-en-features2-fixed-32-dense512+drop0.200000+dense512+drop0.200000+lstm512+drop0.200000-01-1.2196.h5

# Limit TensorFlow to just a single card.
CUDA_VISIBLE_DEVICES=0
# Ensure a stable collation order.
LC_ALL=en_US.utf8

predictions-cs.txt: bible21.txt ${MODEL_CS} train-rnn.py
	CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" LC_ALL="${LC_ALL}" ./train-rnn.py --predict ${MODEL_CS} "$<" > "$@"

predictions-en.txt: biblekjv.txt ${MODEL_EN} train-rnn.py
	CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" LC_ALL="${LC_ALL}" ./train-rnn.py --predict ${MODEL_EN} "$<" > "$@"


vybli-cs: bible21.txt train-rnn.py
	CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" LC_ALL="${LC_ALL}" ulimit -t unlimited && nice -n 19 ./train-rnn.py --train "$@" "$<"

vybli-en: biblekjv.txt train-rnn.py
	CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" LC_ALL="${LC_ALL}" ulimit -t unlimited && nice -n 19 ./train-rnn.py --train "$@" "$<"

bible21.txt: SF_2016-10-10_CZE_CZEB21_(CZECH\ BIBLE,\ PREKLAD\ 21_STOLETI).xml
	#Bible21+-2015-pro-web.pdf
	#pdftotext -eol unix -enc UTF-8 -f 19 -l 1830 -layout "$<" - | sed -e 's/ﬁ\s*/fi/g;s/ﬂ/fl/g;s/˝//g;s/’/i/g;s/»/„/g;s/«/“/g;s|/|–|g;/[\f=×]/ d;s/  */ /g' > "$@"
	egrep '\s*</?(VERS|BIBLEBOOK|CHAPTER)' "$<" |dos2unix |sed -e 's/^\s*//;s|</VERS>||g;s|</CHAPTER>||;s|</BIBLEBOOK>||;s|<BIBLEBOOK bnumber="\([0-9]*\)" bname="\([^"]*\)" bsname="[^"]*">|Kniha \1.: \2\n|;s|<CHAPTER cnumber="\([0-9]*\)">|(\1)|;s|<VERS vnumber="\([0-9]*\)">|\1. |;s/ / /g' > "$@"

biblekjv.txt: SF_2009-01-23_ENG_KJV_(KING\ JAMES\ VERSION).xml
	egrep '\s*</?(VERS|BIBLEBOOK|CHAPTER)' "$<" |dos2unix |sed -e 's/^\s*//;s|</VERS>||g;s|</CHAPTER>||;s|</BIBLEBOOK>||;s|<BIBLEBOOK bnumber="\([0-9]*\)" bname="\([^"]*\)" bsname="[^"]*">|Book \1: \2\n|;s|<CHAPTER cnumber="\([0-9]*\)">|(\1)|;s|<VERS vnumber="\([0-9]*\)">|\1. |;s/ / /g' > "$@"

Bible21+-2015-pro-web.pdf:
	wget 'http://www.bible21.cz/wp-content/uploads/2014/03/Bible21+-2015-pro-web.pdf' -O "$@"


perplexity.png: perplexity.gnuplot perplexity.tsv
	gnuplot "$<"

clean:
	rm -f bible21.txt biblekjv.txt
	#rm -f vybli-cs*.h5 vybli-en*.h5
	rm -f perplexity.png
