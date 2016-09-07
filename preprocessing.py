import pandas
import re
import collections

def generate_content_types(email):
    output = collections.defaultdict(bool)
    check = ['x-world', 'application', 'text', 'text/plain', 'text/html', 'video', 'audio', 'image', 'drawing', 'model', 'multipart', 'x-conference', 'i-world', 'music', 'message', 'x-music', 'www', 'chemical', 'paleovu', 'windows', 'xgl']
    
    for part in email.walk():
        ct = part.get_content_type()
        
        for kind in check:
            output['has_' + kind] |= ct.startswith(kind)
    
    return output

def generate_number_of_spaces(email):
    email = str(email)
    
    return {
        'spaces': email.count(' '),
        'newlines': email.count('\n')
    }

def generate_number_of_images(email):
    output = { 'multipart_number': 0, 'number_of_images': 0 }
    rgx = re.compile('\.(jpeg|jpg|png|gif|bmp)')
    
    for part in email.walk():
        output['multipart_number'] += 1
        
        if part.get_content_type().startswith('image/'):
            output['number_of_images'] += 1
        elif part.get_content_type() == 'text/html' or part.get_content_type() == 'text/plain':
            output['number_of_images'] += len(re.findall(rgx, part.get_payload()))
    
    return output

def generate_contact_numbers(email):
    def normalize(contacts):
        if pandas.isnull(contacts):
            return []
        else:
            contacts = str(contacts)
            contacts = ''.join([x for x in contacts if x not in ['#', '\n', '\t', '\r']])
            contacts = contacts.split(',')
            return [c for c in contacts if c != '']
    
    check = ['to', 'x-to', 'from', 'x-from', 'cc', 'x-cc', 'bcc', 'x-bcc']
    output = {}
    
    # We use -1 to mean that the header was not present in the original email
    for header in check:
        output['people_in_' + header] = -1
    
    for header in email.keys():
        header = header.lower()
        
        if header in check:
            output['people_in_' + header] = len(normalize(email[header]))
    
    return output

def generate_upper_to_lower_case_ratios(email):
    import html2text

    output = { 'title_case_words_to_words_ratio': 0.0, 'upper_case_letters_to_letters_ratio': 0.0 }

    r_words = re.compile(r'\b\w+\b')
    r_upper_words = re.compile(r'\b[A-Z]\w*\b')
    r_letters = re.compile(r'[a-z]')
    r_upper_letters = re.compile(r'[A-Z]')

    for content in email.walk():
        content_type = content.get_content_type()

        if content_type in ('text/plain', 'text/html'):
            if content_type.endswith('plain'):
                body = content.get_payload()
            elif content_type.endswith('html'):
                body = html2text.html2text(content.get_payload())

            total_words = len(r_words.findall(body))
            upper_case_words = len(r_upper_words.findall(body))
            total_letters = len(r_letters.findall(body))
            upper_case_letters = len(r_upper_letters.findall(body))

            # We use -1 to mean that the amount could not be computed because of a problem in the data
            # Notice that this is different than not being computed because of a different format
            if total_words > 0:
                output['title_case_words_to_words_ratio'] = upper_case_words / total_words
            else:
                output['title_case_words_to_words_ratio'] = -1.0

            if total_letters > 0:
                output['upper_case_letters_to_letters_ratio'] = upper_case_letters / total_letters
            else:
                output['upper_case_letters_to_letters_ratio'] = -1.0

    return output

def generate_subject_is_chain(email):
    def get_subject(email):
        try:
            # TODO: check what happens with ?=?
            s = re.search(r'^(fwd|re|fw):', email['subject'], re.IGNORECASE)

            if s is not None:
                return s.group(1).lower()
        except:
            pass

        return None
    
    subject = get_subject(email)
    output = {
        'has_subject': True,
        'is_fwd': False,
        'is_re': False,
        'is_fw': False
    }
    
    if subject is not None:
        output['is_'+subject] = True
    else:
        output['has_subject'] = False

    return output

def generate_number_of_links(email):
    output = { 'number_of_links': 0 }
    r_links = re.compile('(https?|ftps?|mailto|file|data)://')

    for part in email.walk():
        if part.get_content_type().startswith('text'):
            output['number_of_links'] += len(r_links.findall(part.get_payload()))

    return output

def generate_is_mailing_list(email):
    output = {
        'is_mailing_list_by_headers': False,
        'is_mailing_list_by_subject': False,
        'is_mailing_list_by_address': False
    }

    check = ['list-id', 'list-post', 'list-help', 'list-unsubscribe', 'list-owner']

    output['is_mailing_list_by_headers'] = len([x for x in check if x in email.keys()]) > 0
    output['is_mailing_list_by_subject'] = re.search(r'^\[[A-Za-z_\-]{2,}\]') is not None
    # TODO: implement function to detect email address, domain, username, and maybe name.
    #output['is_mailing_list_by_address'] = 

    return output

# Functions which create the output features
# These must always return a dictionary with the same keys for eveyr row, all of which must be non-null
transforms = [
    lambda email: {'length': len(email)},
    generate_content_types,
    generate_number_of_spaces,
    generate_number_of_images,
    generate_contact_numbers,
    generate_upper_to_lower_case_ratios,
    generate_subject_is_chain,
    generate_number_of_links,
    generate_is_mailing_list]

# Process a single row, as received from the pandas.DataFrame iterator
def transform_row(x):
    (index, row) = x

    current = {
        'class': row['class']
    }
    
    # Parse email into Python's class
    row['email'] = email.message_from_string(row['email'])

    # Apply the transform features to the email object
    for function in transforms:
        current.update(function(row['email']))
    
    return current

# WARNING: This check is required to avoid loading the dataset again when creating a new thread
if __name__ == '__main__':
    import multiprocessing
    import email

    # Load dataset
    dataset = pandas.read_msgpack('./data/development.msg', encoding='latin-1')

    # Set up multiprocessing pool
    # WARNING: number of threads should be no more than #cores + 1
    pool = multiprocessing.Pool(4)
    # Generate features for every row
    transformed = pool.map(transform_row, dataset.iterrows())

    # Dump the processed dataset to disk
    preprocessed = pandas.DataFrame(transformed)
    preprocessed.to_msgpack('./data/processed.msg')