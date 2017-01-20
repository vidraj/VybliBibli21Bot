.PHONY: clean

vybli-model-small.h5: bible21-small.txt
	ulimit -t unlimited && nice -n 19 ./train-rnn.py "$@" < "$<"

vybli-model.h5: bible21.txt
	ulimit -t unlimited && nice -n 19 ./train-rnn.py "$@" < "$<"

bible21-small.txt: bible21.txt
	head -c 500000 "$<" > "$@"

bible21.txt: Bible21+-2015-pro-web.pdf
	pdftotext -eol unix -enc UTF-8 -f 19 -l 1830 "$<" - | sed -e 's/ﬁ\s*/fi/g;s/ﬂ/fl/g;s/˝//g' > "$@"

Bible21+-2015-pro-web.pdf:
	wget 'http://www.bible21.cz/wp-content/uploads/2014/03/Bible21+-2015-pro-web.pdf' -O "$@"

clean:
	rm -f bible21.txt bible21-small.txt vybli-model.h5 vybli-model-small.h5


