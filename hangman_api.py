'''The file includes functions used for calling hangman API 
and the guess function 
'''

import json
import requests
import random
import string
import secrets
import time
import re
import collections

import torch
import bidirectional_lstm as bi_lstm
import model_jinna as model_jn
import json


try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# load 2 lstm model 
# 1.one directional lstm
# model = model_jn.LSTMCharModel(vocab_size = 27)
# model.load_state_dict(torch.load('lstm_model.pth'))
# model.eval()

# with open("model_metadata.json", "r") as f:
#     data = json.load(f)
# # Extract specific parts
# char_to_idx = data["char_to_idx"]
# idx_to_char = {int(k): v for k, v in data["idx_to_char"].items()}  # Convert keys to int
# max_len = data["max_len"]

# # 2. bidirectional lstm 
# bi_lstm_model, vocab = bi_lstm.load_model() 


class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None):
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        
        full_dictionary_location = "data/words_250000_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)        
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        
        self.current_dictionary = []
        self.n_gram_dictionary = []
        self.n_gram_backup = []
        self.n_gram_used = 0 
        self.current_word = ''
        
    # @staticmethod
    def determine_hangman_url(self):
        links = ['https://trexsim.com']

        data = {link: 0 for link in links}

        for link in links:

            requests.get(link)

            for i in range(10):
                s = time.time()
                requests.get(link)
                data[link] = time.time() - s

        link = sorted(data.items(), key=lambda x: x[1])[0][0]
        link += '/trexsim/hangman'
        return link

    def guess(self, word): # word input example: "_ p p _ e "
        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        clean_word = word[::2].replace("_",".")
        word_for_model = word[::2]
        # print('word_for_model is '+word_for_model)

        # find length of passed word
        len_word = len(clean_word)
        
        # for the first few letters (i.e. first 1 letters for word with lengthen <= 4, 
        # first 2 for lengthen = 5, first 3 for lengthen >= 15 )
        if len_word*0.8 <= clean_word.count("."):
            return self.most_common_letter_guess(clean_word)
        else:
            ###### N_gram
            n_gram_min = max(len_word-3,3)
            # if len(self.n_gram_dictionary) == 0:
            #     n_gram_dictionary = self.build_n_gram_dict(clean_word, min(n_gram_min,len(clean_word)))
            #     self.n_gram_dictionary = n_gram_dictionary
            # else:
            #     n_gram_dictionary = self.n_gram_dictionary
            if len(self.n_gram_dictionary) == 0 and self.n_gram_used == 0:
                n_gram_dictionary = self.build_n_gram_dict(clean_word, min(n_gram_min,len(clean_word)))
                n_gram_backup = self.build_n_gram_dict(clean_word, min(3,len(clean_word)))
                self.n_gram_dictionary = n_gram_dictionary
                self.n_gram_backup = n_gram_backup
                self.n_gram_used = 1
            # elif len(self.n_gram_dictionary) == 0 and self.n_gram_used == 1:
            #     n_gram_dictionary = self.n_gram_backup
                # print('used n_gram_backup with lengthen: ' + str(len(n_gram_dictionary)))
            else:
                n_gram_dictionary = self.n_gram_dictionary
            print('len(self.n_gram_dictionary) is ' + str(len(self.n_gram_dictionary)))
            print('len(self.n_gram_backup) is ' + str(len(self.n_gram_backup)))
            print('len(n_gram_dictionary) is ' + str(len(n_gram_dictionary)))

            target_grams = self.get_ngrams(clean_word,min(n_gram_min,len(clean_word)))
            
            ###### smaller but more accurate N_gram dictionary 
            # collect all the possible letters from n_gram match
            matched_chars_ngram,matched_ngrams = self.matched_letter(target_grams,n_gram_dictionary)
            self.n_gram_dictionary = matched_ngrams
            # print('len(self.n_gram_dictionary) is ' + str(len(self.n_gram_dictionary)))
            print('some matched grams are: ')
            print(matched_ngrams[:3] + matched_ngrams[-3:])
            # print('number of matched_ngrams are ' + str(len(matched_ngrams)))
            
            # count occurrence of all characters in possible word matches
            full_dict_string = matched_chars_ngram
            
            c = collections.Counter(full_dict_string)
            sorted_letter_count = c.most_common() 
            print('letters predicted by n_grams are: ')
            filtered_most_common_by_ngram = [(char, prob) for char, prob in sorted_letter_count if char not in self.guessed_letters]
            print(filtered_most_common_by_ngram)   
            
            ###### switch to larger N_gram dictionary 
            if len(filtered_most_common_by_ngram) == 0:
                n_gram_dictionary = self.n_gram_backup
                target_grams = self.get_ngrams(clean_word,min(3,len(clean_word)-3))
                print('check target_grams: ')
                print(target_grams)
               
                matched_chars_ngram,matched_ngrams = self.matched_letter(target_grams,n_gram_dictionary)
                self.n_gram_backup = matched_ngrams
                full_dict_string = matched_chars_ngram
                c = collections.Counter(full_dict_string)
                sorted_letter_count = c.most_common() 
                filtered_most_common_by_ngram = [(char, prob) for char, prob in sorted_letter_count if char not in self.guessed_letters]
                print('check for the new list!:')
                print('some matched grams are: ')
                print(matched_ngrams[:3] + matched_ngrams[-3:])
                print('len(matched_ngrams) is '+str(len(matched_ngrams)))
                print(filtered_most_common_by_ngram)
                


            final_list = filtered_most_common_by_ngram

            # check model result first          
            # print('top 5 letters predicted by birectional lstm are: ')
            # most_common_by_bimodel = bi_lstm.predict_missing(bi_lstm_model, vocab, word_for_model)
            # filtered_most_common_by_bimodel = [(char, prob) for char, prob in most_common_by_bimodel if char not in self.guessed_letters]
            # print(filtered_most_common_by_bimodel[:7])
            
            # if len_word*0.2 >= clean_word.count("."): 
            # if clean_word.count(".") <= 2: 
            #     print('top 10 letters predicted by one directional lstm are: ')
            #     most_common_by_model = model_jn.get_ranked_letter_probs(model, word_for_model, char_to_idx, idx_to_char, max_len)
            #     filtered_most_common_by_model = [(char, prob) for char, prob in most_common_by_model if char not in self.guessed_letters]
            #     print(filtered_most_common_by_model[:10])

            #     print('--------------combined results are-------------------')
            #     combined = self.combine_and_filter_predictions(most_common_by_model, sorted_letter_count, self.guessed_letters, 0.17)
            #     print(combined[:10])
            #     print('-----------------------------------------------------')
            #     print()
                
            #     final_list = combined


            guess_letter = '!'
            
            # return most frequently occurring letter in all possible words that hasn't been guessed yet
            for letter,instance_count in final_list:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break
                
            # if no word matches in training dictionary, default back to ordering of full dictionary
            if guess_letter == '!':
                sorted_letter_count = self.full_dictionary_common_letter_sorted
                for letter,instance_count in sorted_letter_count:
                    if letter not in self.guessed_letters:
                        guess_letter = letter
                        break 
            
            return guess_letter
    
    def combine_and_filter_predictions(self, lstm_result, ngram_result, guessed_letters, alpha):
        """
        Combine LSTM and N-gram results with weighting, and remove guessed letters.

        Parameters:
        - lstm_result: list of (char, prob) from LSTM, e.g. [('a', 0.5), ('b', 0.3), ...]
        - ngram_result: list of (char, count) from N-gram, e.g. [('a', 50000), ('b', 45000), ...]
        - guessed_letters: set or list of already guessed letters, e.g. {'a', 'e'}
        - alpha: weight for LSTM, (1 - alpha) is the weight for N-gram

        Returns:
        - combined: list of (char, combined_score), sorted by score descending
        """
        # Normalize ngram counts to probabilities
        total_ngram = sum(count for _, count in ngram_result)
        ngram_probs = {char: count / total_ngram for char, count in ngram_result}

        # Convert LSTM result to dict
        lstm_probs = dict(lstm_result)

        # Combine scores
        all_chars = set(lstm_probs) | set(ngram_probs)
        combined = {}
        for char in all_chars:
            if char in guessed_letters:
                continue
            lstm_p = lstm_probs.get(char, 0)
            ngram_p = ngram_probs.get(char, 0)
            combined[char] = alpha * lstm_p + (1 - alpha) * ngram_p

        # Sort by combined score
        sorted_combined = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_combined


    def matched_letter(self, target_grams, ngrams_dict):
        matched_chars = [] 
        matched_grams = []
        for target_gram in target_grams:
            pattern_len = len(target_gram)
            for ngram in ngrams_dict:
                if len(ngram) != pattern_len:
                    continue
                pattern = f"^{target_gram}$"
                if re.match(pattern,ngram):
                    letters = [ngc for pc, ngc in zip(target_gram, ngram) if pc == '.']
                    matched_chars.extend(letters)
                    matched_grams.append(ngram)
        return matched_chars, matched_grams
    
    def build_n_gram_dict(self,target_word, min_n):
        '''build n_gram dictionary with n ranges from min_n to len(clean_word)'''
        full_dictionary = self.full_dictionary
        n_gram_dictionary = []
        for word in full_dictionary:
            for n in range(min_n, 1+min(len(word),len(target_word))):
                pieces = [word[i:i+n] for i in range(len(word)-n+1)]
                n_gram_dictionary.extend(pieces)
        return n_gram_dictionary
    
    def get_ngrams(self,word,min_n):
        """
        Returns a list of n-grams (n from min_n to len(clean_word)) from clean_word
        that contain at least one '.' character but not all '.'
        """
        ngrams = []
        length = len(word)
        
        for n in range(min_n, length + 1):  # n from 2 to len(clean_word)
            for i in range(length - n + 1):
                piece = word[i:i+n]
                if '.' in piece and any(c != '.' for c in piece):
                    ngrams.append(piece)
        
        return ngrams

    def most_common_letter_guess(self,word):
        current_dictionary = self.current_dictionary
        new_dictionary = []
        len_word = len(word)

        # iterate through all of the words in the old plausible dictionary
        for dict_word in current_dictionary:
            # continue if the word is not of the appropriate length
            # if len(dict_word) != len_word:
            #     continue
            if len(dict_word) > len_word + 3 or len(dict_word) < len_word - 3:
                continue
                
            # if dictionary word is a possible match then add it to the current dictionary
            if re.match(word,dict_word):
                new_dictionary.append(dict_word)
        
        # overwrite old possible words dictionary with updated version
        self.current_dictionary = new_dictionary
        
        # count occurrence of all characters in possible word matches
        full_dict_string = "".join(new_dictionary)
        
        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()                   
        
        guess_letter = '!'
        
        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        for letter,instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                guess_letter = letter
                break
            
        # if no word matches in training dictionary, default back to ordering of full dictionary
        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter,instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break 
        
        return guess_letter


    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################
    
    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
                
    def start_game(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
        self.n_gram_dictionary = []
        self.n_gram_used = 0
                         
        response = self.request("/new_game", {"practice":practice})
        if response.get('status')=="approved":
            game_id = response.get('game_id')
            word = response.get('word')
            self.current_word = word
            tries_remains = response.get('tries_remains')
            if verbose:
                print("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id, tries_remains, word))
            while tries_remains>0:
                # get guessed letter from user code
                guess_letter = self.guess(word)
                    
                # append guessed letter to guessed letters field in hangman object
                self.guessed_letters.append(guess_letter)
                if verbose:
                    print("Guessing letter: {0}".format(guess_letter))
                    
                try:    
                    res = self.request("/guess_letter", {"request":"guess_letter", "game_id":game_id, "letter":guess_letter})
                except HangmanAPIError:
                    print('HangmanAPIError exception caught on request.')
                    continue
                except Exception as e:
                    print('Other exception caught on request.')
                    raise e
               
                if verbose:
                    print("Sever response: {0}".format(res))
                status = res.get('status')
                tries_remains = res.get('tries_remains')
                if status=="success":
                    if verbose:
                        print("Successfully finished game: {0}".format(game_id))
                    return True
                elif status=="failed":
                    reason = res.get('reason', '# of tries exceeded!')
                    if verbose:
                        print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                    return False
                elif status=="ongoing":
                    word = res.get('word')
        else:
            if verbose:
                print("Failed to start a new game")
        return status=="success"
        
    def my_status(self):
        return self.request("/my_status", {})
    
    def request(
            self, path, args=None, post_args=None, method=None):
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"

        # Add `access_token` to post_args or args if it has not already been
        # included.
        if self.access_token:
            # If post_args exists, we assume that args either does not exists
            # or it does not need `access_token`.
            if post_args and "access_token" not in post_args:
                post_args["access_token"] = self.access_token
            elif "access_token" not in args:
                args["access_token"] = self.access_token

        time.sleep(0.2)

        num_retry, time_sleep = 50, 2
        for it in range(num_retry):
            try:
                response = self.session.request(
                    method or "GET",
                    self.hangman_url + path,
                    timeout=self.timeout,
                    params=args,
                    data=post_args,
                    verify=False
                )
                break
            except requests.HTTPError as e:
                response = json.loads(e.read())
                raise HangmanAPIError(response)
            except requests.exceptions.SSLError as e:
                if it + 1 == num_retry:
                    raise
                time.sleep(time_sleep)

        headers = response.headers
        if 'json' in headers['content-type']:
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_str = parse_qs(response.text)
            if "access_token" in query_str:
                result = {"access_token": query_str["access_token"][0]}
                if "expires" in query_str:
                    result["expires"] = query_str["expires"][0]
            else:
                raise HangmanAPIError(response.json())
        else:
            raise HangmanAPIError('Maintype was not text, or querystring')

        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result
    
class HangmanAPIError(Exception):
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""

        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.type:
                    self.type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = result

        Exception.__init__(self, self.message)