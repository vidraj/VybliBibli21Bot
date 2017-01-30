#!/usr/bin/env python3

import sys
import regex as re
from time import sleep
import tweepy
from credentials import consumer_key, consumer_secret, access_token, access_token_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
twitter = tweepy.API(auth, retry_count=3, retry_delay=10) # TODO add the errors to retry to retry_errors


def calculate_tweet_score(text):
	if len(text) > 140:
		# The text is not tweetable.
		return -1
	
	# Let's assume that longer tweets are better.
	score = len(text)
	
	# Check that the last line ends with a dot, question mark, exclamation mark, or that+endquote.
	match_end_interpunction = re.compile(u'[.?!]["\']?$', flags=re.VERSION1)
	if match_end_interpunction.search(text):
		score += 200
	
	# First line begins with a capital letter.
	match_start_upper = re.compile(u'^\\d+\\.\\s+\\p{Lu}', flags=re.VERSION1)
	if match_start_upper.match(text):
		score += 200
	
	# Starts with '(\d)' or 'Kniha \d.:'
	if re.match(u'^\\(\\d+\\)', text, flags=re.VERSION1):
		score += 260
	elif re.match(u'^(Kniha|Book) \\d+\\.?: ', text, flags=re.VERSION1):
		score += 360
	
	# does NOT end with '(\d)' or 'Kniha \d.:'
	if re.search(u'\\(\\d+\\)$', text, flags=re.VERSION1):
		score -= 270
	elif re.match(u'(Kniha|Book) \\d+\\.?: [^\n]+$', text, flags=re.VERSION1):
		score -= 370
	
	# TODO or contain these ^^ inside.
	# TODO parentheses are balanced
	# TODO quotes (both single and double) are balanced and closed
	# TODO verse numbers are in a sequence
	
	return score



# Read NULL-terminated generated text from STDIN, split into Unicode strings. Ignore the segment after the last terminating NULL
tweets_binary = sys.stdin.buffer.read()
tweets_texts = list(map(lambda tweet: tweet.decode('utf-8'), tweets_binary.split(b'\x00')))[:-1]
chosen_tweets = []

#tweets_text = sys.stdin.read()
for tweets_text in tweets_texts:
	# Filter all verses (lines) that are complete – start with a verse number and end with a newline.
	# Do NOT use a $ instead of the lookahead, because that matches the end of the whole string as well, therefore giving potentially incomplete lines.
	verse_filter = re.compile(u'^(\\d+\\. [^\n]+|\\(\\d+\\)|Kniha \\d+\\.: [^\n]+|Book \\d+: [^\n]+)(?=\n)', flags=(re.VERSION1|re.MULTILINE))
	verses = verse_filter.findall(tweets_text)

	#print('Number of verses retrieved: %d' % len(verses))
	#for verse in verses:
	#	print(verse, "\n\n---------------\n\n")


	best_tweet = "ERROR! This is not a biblical tweet!"
	best_tweet_score = 0
	for first_verse in range(len(verses)):
		for last_verse in range(first_verse + 1, len(verses) + 1):
			tweet = "\n".join(verses[first_verse:last_verse])
			score = calculate_tweet_score(tweet)
			if score > best_tweet_score:
				# Found a better tweet, remember it.
				best_tweet = tweet
				best_tweet_score = score
			elif score < 0:
				# The text is too long; adding another verse is not going to help.
				# Try from a different starting point.
				break

	if best_tweet_score == 0:
		sys.stderr.write('No tweetable text found in this stanza of length %d!\n' % len(tweets_text))
		# Advance to the next tweet immediately.
		continue

	assert(len(best_tweet) > 0)
	assert(not re.fullmatch('\\s*', best_tweet, flags=re.VERSION1))
	
	#print("Would tweet this text of len %d with score %d:\n%s\n----------" % (len(best_tweet), best_tweet_score, best_tweet))
	chosen_tweets.append(best_tweet)


print("Found %d tweets." % len(chosen_tweets))
chosen_tweets = chosen_tweets[:31] # Only tweet the first 31 things.
#print("Found %d tweets." % len(chosen_tweets))

for tweet in chosen_tweets:
	twitter.update_status(tweet)
	sleep(2 * 600) # 2×, because we'll be tweeting Czech and English texts interleaved with each other.
	#print("Would tweet this text of len %d:\n%s\n----------" % (len(tweet), tweet))
	#sleep(10) # TODO comment
