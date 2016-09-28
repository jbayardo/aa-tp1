import collections
import re


def generate_content_type(email):
    features = collections.defaultdict(int)
    find_image_extension_regex = re.compile('\.(jpeg|jpg|png|gif|bmp)')

    content_type_prefixes = ['x-world', 'application', 'text', 'text/plain', 'text/html', 'video', 'audio', 'image', 'drawing', 'model',
             'multipart', 'x-conference', 'i-world', 'music', 'message', 'x-music', 'www', 'chemical', 'paleovu',
             'windows', 'xgl']

    features['number_of_multiparts'] = 0
    features['number_of_mentioned_images'] = 0

    for prefix in content_type_prefixes:
        features['number_of_' + prefix + '_multiparts'] = 0

    for part in email.walk():
        content_type = part.get_content_type()

        # This is just the total amount of parts in the email
        features['number_of_multiparts'] += 1

        for prefix in content_type_prefixes:
            if content_type.startswith(prefix):
                features['number_of_' + prefix + '_multiparts'] += 1

        if content_type in ('text/html', 'text/plain'):
            # This counts referenced images through links or simply text
            features['number_of_mentioned_images'] += len(re.findall(find_image_extension_regex, part.get_payload()))

    features['number_of_mentioned_images'] += features['number_of_image_multiparts']

    return features


def generate_email_counts(email):
    import pandas

    def extract_target_list(contacts):
        if pandas.isnull(contacts):
            return []
        else:
            contacts = str(contacts)
            contacts = ''.join([x for x in contacts if x not in ['#', '\n', '\t', '\r']])
            contacts = contacts.split(',')
            return [c for c in contacts if c != '']

    payload = str(email)

    features = {
        'number_of_spaces': payload.count(' '),
        'number_of_newlines': payload.count('\n'),
        'length_of_body': len(email.get_payload()),
        'number_of_spaces_in_body': email.get_payload().count(' '),
        'number_of_newlines_in_body': email.get_payload().count('\n'),
        'number_of_question_marks_in_body': email.get_payload().count('?'),
        'number_of_exclamation_marks_in_body': email.get_payload().count('!'),
    }

    if email['subject'] is not None:
        features.update({
            'length_of_subject': len(email['subject']),
            'number_of_spaces_in_subject': email['subject'].count(' '),
            'number_of_question_marks_in_subject': email['subject'].count('?'),
            'number_of_exclamation_marks_in_subject': email['subject'].count('!'),
        })
    else:
        features.update({
            'length_of_subject': -1,
            'number_of_spaces_in_subject': -1,
            'number_of_question_marks_in_subject': -1,
            'number_of_exclamation_marks_in_subject': -1,
        })

    people_information_headers = ['to', 'x-to', 'from', 'x-from', 'cc', 'x-cc', 'bcc', 'x-bcc']

    for header in people_information_headers:
        if header in email.keys():
            features['number_of_people_in_' + header] = len(extract_target_list(email[header]))
        else:
            features['number_of_people_in_' + header] = -1

    return features


def generate_upper_to_lower_case_ratios(email):
    import html2text

    features = collections.defaultdict(int)

    regexes = {
        'words': re.compile(r'\w+'),
        'lower_case_words': re.compile(r'[a-z0-9]+'),
        'upper_case_words': re.compile(r'[A-Z0-9]+'),
        'start_with_lower_case_words': re.compile(r'\b[a-z]\w+'),
        'start_with_upper_case_words': re.compile(r'\b[A-Z]\w+'),
        'letters': re.compile(r'[a-zA-Z]'),
        'lower_case_letters': re.compile(r'[a-z]'),
        'upper_case_letters': re.compile(r'[A-Z]'),
        'upper_case_sequence': re.compile(r'[A-Z]+'),
        'lower_case_sequence': re.compile(r'[a-z]+'),
        'digits': re.compile(r'[0-9]'),
        'digit_sequence': re.compile(r'[0-9]+'),
        'money_count': re.compile(r'(\$|€|£)'),
        'money_sequence': re.compile(r'(\$+|€+|£+)')
    }

    for key in regexes:
        features[key + '_matches'] = -1
        features[key + '_shortest_match'] = -1
        features[key + '_longest_match'] = -1

    # TODO: finish
    symbol_counts = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '=', '-', '{', '}', '|', '"', '\'', '\\', '[', ']', '/', '?', '>', '<', ',', ':', ';']

    for content in email.walk():
        content_type = content.get_content_type()

        if content_type in ('text/plain', 'text/html'):
            body = content.get_payload()

            if content_type.endswith('html'):
                body = html2text.html2text(body)

            for key in regexes:
                matches = regexes[key].findall(body)
                lengths = [len(x) for x in matches]

                features[key + '_matches'] += len(matches)

                if len(matches) > 0:
                    features[key + '_shortest_match'] = min(min(lengths), features[key + '_shortest_match'])
                    features[key + '_longest_match'] = max(max(lengths), features[key + '_longest_match'])

            # We use -1 to mean that the amount could not be computed because of a problem in the data
            # Notice that this is different than not being computed because of a different format
            if features['words_matches'] > 0:
                features['ratio_of_lower_case_words'] = features['lower_case_words_matches'] / features['words_matches']
                features['ratio_of_upper_case_words'] = features['upper_case_words_matches'] / features['words_matches']
                features['ratio_of_start_with_lower_case_words'] = features['start_with_lower_case_words_matches'] / features['words_matches']
                features['ratio_of_start_with_upper_case_words'] = features['start_with_upper_case_words_matches'] / features['words_matches']
            else:
                features['ratio_of_lower_case_words'] = -1.0
                features['ratio_of_upper_case_words'] = -1.0
                features['ratio_of_start_with_lower_case_words'] = -1.0
                features['ratio_of_start_with_upper_case_words'] = -1.0

            if features['lower_case_words_matches'] > 0:
                features['ratio_of_lower_case_to_start_with_lower_case_words'] = features['start_with_lower_case_words_matches'] / features['lower_case_words_matches']
            else:
                features['ratio_of_lower_case_to_start_with_lower_case_words'] = -1.0

            if features['upper_case_words_matches'] > 0:
                features['ratio_of_upper_case_to_start_with_upper_case_words'] = features['start_with_upper_case_words_matches'] / features['upper_case_words_matches']
            else:
                features['ratio_of_upper_case_to_start_with_upper_case_words'] = -1.0

            if features['letters_matches'] > 0:
                features['ratio_of_lower_case_letters'] = features['lower_case_letters_matches'] / features['letters_matches']
                features['ratio_of_upper_case_letters'] = features['upper_case_letters_matches'] / features['letters_matches']
            else:
                features['ratio_of_lower_case_letters'] = -1.0
                features['ratio_of_upper_case_letters'] = -1.0

            break

    return features


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

    features = {
        'has_subject': True,
        'is_fwd': False,
        'is_re': False,
        'is_fw': False
    }

    if subject is not None:
        features['is_' + subject] = True
    else:
        features['has_subject'] = False

    return features


def generate_number_of_links(email):
    features = { 'number_of_links': 0 }

    r_links = re.compile('(https?|ftps?|mailto|file|data)://')

    for part in email.walk():
        if part.get_content_type().startswith('text'):
            features['number_of_links'] += len(r_links.findall(part.get_payload()))

    return features


def generate_is_mailing_list(email):
    features = {
        'is_mailing_list_by_headers': False,
        'is_mailing_list_by_subject': False,
        'is_mailing_list_by_address': False
    }

    check = ['list-id', 'list-post', 'list-help', 'list-unsubscribe', 'list-owner']

    features['is_mailing_list_by_headers'] = len([x for x in check if x in email.keys()]) > 0
    features['is_mailing_list_by_subject'] = re.search(r'^\[[A-Za-z_\-]{2,}\]', email['subject']) is not None if email['subject'] is not None else False
    # TODO: implement function to detect email address, domain, username, and maybe name.
    # output['is_mailing_list_by_address'] =

    return features
