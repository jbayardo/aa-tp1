import pandas as pd
import re
import multiprocessing
import collections
from load_development_dataset import df 


def generate_content_types(row):
    email = row['email']
    output = collections.defaultdict(bool)
    check = ['x-world', 'application', 'text', 'text/plain', 'text/html', 'video', 'audio', 'image', 'drawing', 'model', 'multipart', 'x-conference', 'i-world', 'music', 'message', 'x-music', 'www', 'chemical', 'paleovu', 'windows', 'xgl']
    
    for part in row['email'].walk():
        ct = part.get_content_type()
        
        for kind in check:
            output['has_' + kind] |= ct.startswith(kind)
    
    return output

def generate_number_of_spaces(row):
    email = str(row['email'])
    
    return {
        'spaces': email.count(' '),
        'newlines': email.count('\n')
    }

def generate_number_of_images(row):
    email = row['email']
    output = { 'multipart_number': 0, 'number_of_images': 0 }
    rgx = re.compile('\.(jpeg|jpg|png|gif|bmp)')
    
    for part in email.walk():
        output['multipart_number'] += 1
        
        if part.get_content_type().startswith('image/'):
            output['number_of_images'] += 1
        elif part.get_content_type() == 'text/html' or part.get_content_type() == 'text/plain':
            output['number_of_images'] += len(re.findall(rgx, part.get_payload()))
    
    return output

def generate_contact_numbers(row):
    def normalize(contacts):
        if pd.isnull(contacts):
            return []
        else:
            contacts = str(contacts)
            contacts = ''.join([x for x in contacts if x not in ['#', '\n', '\t', '\r']])
            contacts = contacts.split(',')
            return [c for c in contacts if c != '']
    
    check = ['to', 'x-to', 'from', 'x-from', 'cc', 'x-cc', 'bcc', 'x-bcc']
    output = {}
    
    for header in check:
        output[header] = 0
    
    for header in row['email'].keys():
        header = header.lower()
        
        if header in check:
            output[header] = len(normalize(row['email'][header]))
    
    return output

def generate_upper_to_lower_case_ratios(row):
    email = row['email']
    output = collections.defaultdict(float)

    r_words = re.compile(r'\b\w+\b')
    r_upper_words = re.compile(r'\b[A-Z]\w*\b')
    r_letters = re.compile(r'\[a-z]')
    r_upper_letters = re.compile(r'[A-Z]')

    for content in email.walk():
        content_type = content.get_content_type()

        if content_type[:4] in ['text', 'html']:
            if content_type.startswith('text/'):
                body = content.get_payload()
            elif content_type.startswith('html/'):
                body = html2text(content.get_payload())

            totat_words = len(r_words.findall(body))
            upper_case_words = len(r_upper_words.findall(body))
            totat_letters = len(r_words.findall(body))
            upper_case_letters = len(r_upper_words.findall(body))
            output['title_case_words_to_words_ratio'] = upper_case_words / total_words
            output['upper_case_letters_to_letters_ratio'] = upper_case_letters / total_letters

    return output

def generate_subject_features(row):
    return {}

# Functions which create the output features
transforms = [
    lambda row: {'class': row['class']},
    lambda row: {'length': len(row['email'])},
    generate_content_types,
    generate_number_of_spaces,
    generate_number_of_images,
    generate_contact_numbers,
    generate_upper_to_lower_case_ratios,
    generate_subject_features]

# Set up thread pool
def transform_row(x):
    (index, row) = x
    current = {}
    
    for function in transforms:
        current.update(function(row))
    
    return current

print("Processing")
pool = multiprocessing.Pool(2)

# Create dataframe
transformed = pool.map(transform_row, df.iterrows())

print("Done!")

try:
    del ds
except:
    pass

ds = pd.DataFrame(transformed)
