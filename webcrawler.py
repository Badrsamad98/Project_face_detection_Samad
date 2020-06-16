import requests
import re
import json
import time
import os
import logging
from pathlib import Path
import csv
from requests.exceptions import HTTPError

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def search(family, config, p, max_results=None):
    for index, keywords in enumerate(family):
        url = 'https://duckduckgo.com/'
        params = {
            'q': keywords
        }

        logger.debug("Hitting DuckDuckGo for Token")

        #   First make a request to above URL, and parse out the 'vqd'
        #   This is a special token, which should be used in the subsequent request
        res = requests.post(url, data=params)
        searchObj = re.search(r'vqd=([\d-]+)\&', res.text, re.M | re.I)

        if not searchObj:
            logger.error("Token Parsing Failed !")
            return -1

        logger.debug("Obtained Token")

        headers = {
            'authority': 'duckduckgo.com',
            'accept': 'application/json, text/javascript, */*; q=0.01',
            'sec-fetch-dest': 'empty',
            'x-requested-with': 'XMLHttpRequest',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-mode': 'cors',
            'referer': 'https://duckduckgo.com/',
            'accept-language': 'en-US,en;q=0.9',
        }

        SAVE_FOLDER = os.getcwd() + '/downloads2/' + '/Family_' + str(p) + '/'

        params = (
            ('l', 'us-en'),
            ('o', 'json'),
            ('q', keywords),
            ('vqd', searchObj.group(1)),
            ('f', ',,,'),
            ('p', '1'),
            ('v7exp', 'a'),
        )

        requestUrl = url + "i.js"

        logger.debug("Hitting Url : %s", requestUrl)
        # The folders containing the images
        folder_path = ''
        if index == 0:
            folder_path = SAVE_FOLDER+file_name_formatter(keywords)+'_father'
            Path(folder_path
                 ).mkdir(parents=True, exist_ok=True)
        elif index == 1:
            folder_path = SAVE_FOLDER+file_name_formatter(keywords)+'_mother'
            Path(folder_path
                 ).mkdir(parents=True, exist_ok=True)
        else:
            folder_path = SAVE_FOLDER+file_name_formatter(keywords)+'_daughter'
            Path(folder_path
                 ).mkdir(parents=True, exist_ok=True)
        # Here we get the Url, we find the images with the dimensions written in config.txt
        j = 0
        imagelinks = []

        while True:
            while True:
                try:
                    res = requests.get(
                        requestUrl, headers=headers, params=params)
                    data = json.loads(res.text)
                    break
                except ValueError as e:
                    logger.debug(
                        "Hitting Url Failure - Sleep and Retry: %s", requestUrl)
                    time.sleep(1)
                    continue

            logger.debug("Hitting Url Success : %s", requestUrl)

            for k in data["results"]:
                if(j == int(config[2].rstrip())):
                    break
                if(k['height'] > int(config[0].rstrip()) and k['width'] > int(config[1].rstrip())):
                    image = k['image']
                    imagelinks.append(image)
                    j += 1

            print(f'found {len(imagelinks)} images')
            print('Start downloading...')
            print(imagelinks)

            for i, imagelink in enumerate(imagelinks):
                # open image link and save as file
                while True:
                    try:
                        response = requests.get(imagelink)
                        # If the response was successful, no Exception will be raised
                        response.raise_for_status()
                    except HTTPError as http_err:
                        print(f'HTTP error occurred: {http_err}')  # Python 3.6
                    except Exception as err:
                        print(f'Other error occurred: {err}')  # Python 3.6
                    else:
                        print('Success!')
                    break

                imagename = Path(folder_path + '/' +
                                 file_name_formatter(keywords) + str(i+1) + '.jpg')
                with open(imagename, 'wb') as file:
                    file.write(response.content)

                linkname = Path(folder_path + '/' +
                                file_name_formatter(keywords) + str(i+1) + '.csv')

                with open(linkname, 'w', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=';',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(
                        ['Filename', file_name_formatter(keywords) + str(i+1) + '.jpg'])
                    spamwriter.writerow(
                        ['Name', keywords.rstrip()])
                    spamwriter.writerow(
                        ['Father', family[0].rstrip()])
                    spamwriter.writerow(
                        ['Mother', family[1].rstrip()])
                    spamwriter.writerow(
                        ['Daughter', family[2].rstrip()])
                    spamwriter.writerow(
                        ['Kin relation', 'Father' if index == 0 else ('Mother' if index == 1 else 'Daughter')])
                    spamwriter.writerow(
                        ['Gender', 'Male' if index == 0 else 'Female'])
                    spamwriter.writerow(
                        ['URL', imagelink])

            print('Done')

            if(j == int(config[2].rstrip())):
                break
            # find the next page with more pictures
            if "next" not in data:
                logger.debug("No Next Page - Exiting")
                break

            requestUrl = url + data["next"]


def file_name_formatter(keywords):
    words = keywords.rstrip().split(" ")
    result = ""
    for word in words:
        result += word + "_"
    result = result[:-1]
    return result


# Creating the text files
inputFile = open('input.txt', 'r')
lines = inputFile.readlines()

configFile = open('config.txt', 'r')
config = configFile.readlines()
# Searching for the names in the input file
family_number = 97
family = []
for line in lines:
    if line.rstrip() == 'begin':
        if(family_number != 0):
            search(family, config, family_number)
        family = []
        family_number += 1
        Path(os.getcwd() + '/downloads2' + '/Family_' + str(family_number)
             ).mkdir(parents=True, exist_ok=True)
    else:
        family.append(line)
