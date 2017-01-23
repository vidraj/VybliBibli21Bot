.PHONY: clean

vybli-model-small.h5: bible21-small.txt
	ulimit -t unlimited && nice -n 19 ./train-rnn.py "$@" < "$<"

vybli-model.h5: bible21.txt
	ulimit -t unlimited && nice -n 19 ./train-rnn.py "$@" < "$<"

bible21-small.txt: bible21.txt
	head -c 10000 "$<" > "$@"

bible21.txt: SF_2016-10-10_CZE_CZEB21_(CZECH\ BIBLE,\ PREKLAD\ 21_STOLETI).xml
	#Bible21+-2015-pro-web.pdf
	#pdftotext -eol unix -enc UTF-8 -f 19 -l 1830 -layout "$<" - | sed -e 's/ﬁ\s*/fi/g;s/ﬂ/fl/g;s/˝//g;s/’/i/g;s/»/„/g;s/«/“/g;s|/|–|g;/[\f=×]/ d;s/  */ /g' > "$@"
	egrep '\s*</?(VERS|BIBLEBOOK|CHAPTER)' "$<" |sed -e 's/^\s*//;s|</VERS>||g;s|</CHAPTER>||;s|</BIBLEBOOK>||;s|<BIBLEBOOK bnumber="\([0-9]*\)" bname="\([^"]*\)" bsname="[^"]*">|Kniha \1.: \2\n|;s|<CHAPTER cnumber="\([0-9]*\)">|(\1)|;s|<VERS vnumber="\([0-9]*\)">|\1. |;s/ / /g' > "$@"

Bible21+-2015-pro-web.pdf:
	wget 'http://www.bible21.cz/wp-content/uploads/2014/03/Bible21+-2015-pro-web.pdf' -O "$@"

clean:
	rm -f bible21.txt bible21-small.txt vybli-model.h5 vybli-model-small.h5
	#rm -f vybli-checkpoint-*.h5


