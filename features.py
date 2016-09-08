import collections
import re


def generate_content_type(email):
    features = collections.defaultdict(int)
    find_image_extension_regex = re.compile('\.(jpeg|jpg|png|gif|bmp)')

    content_type_prefixes = ['x-world', 'application', 'text', 'text/plain', 'text/html', 'video', 'audio', 'image', 'drawing', 'model',
             'multipart', 'x-conference', 'i-world', 'music', 'message', 'x-music', 'www', 'chemical', 'paleovu',
             'windows', 'xgl']

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

    people_information_headers = ['to', 'x-to', 'from', 'x-from', 'cc', 'x-cc', 'bcc', 'x-bcc']

    for header in people_information_headers:
        if header in email.keys():
            features['number_of_people_in_' + header] = len(extract_target_list(email[header]))

    return features


def generate_upper_to_lower_case_ratios(email):
    import html2text

    features = {}
    r_words = re.compile(r'\w+')
    r_lower_case_words = re.compile(r'[a-z0-9]+')
    r_upper_case_words = re.compile(r'[A-Z0-9]+')
    r_start_with_lower_case_words = re.compile(r'[a-z]\w+')
    r_start_with_upper_case_words = re.compile(r'[A-Z]\w+')
    r_letters = re.compile(r'[a-zA-Z]')
    r_lower_case_letters = re.compile(r'[a-z]')
    r_upper_case_letters = re.compile(r'[A-Z]')

    for content in email.walk():
        content_type = content.get_content_type()

        if content_type in ('text/plain', 'text/html'):
            if content_type.endswith('plain'):
                body = content.get_payload()
            elif content_type.endswith('html'):
                body = html2text.html2text(content.get_payload())

            total_words = len(r_words.findall(body))
            total_lower_case_words = len(r_lower_case_words.findall(body))
            total_upper_case_words = len(r_upper_case_words.findall(body))
            total_start_with_lower_case_words = len(r_start_with_lower_case_words.findall(body))
            total_start_with_upper_case_words = len(r_start_with_upper_case_words.findall(body))
            total_letters = len(r_letters.findall(body))
            total_lower_case_letters = len(r_lower_case_letters.findall(body))
            total_upper_case_letters = len(r_upper_case_letters.findall(body))

            # We use -1 to mean that the amount could not be computed because of a problem in the data
            # Notice that this is different than not being computed because of a different format
            features['number_of_words'] = total_words
            features['number_of_lower_case_words'] = total_lower_case_words
            features['number_of_upper_case_words'] = total_upper_case_words
            features['number_of_start_with_lower_case_words'] = total_start_with_lower_case_words
            features['number_of_start_with_upper_case_words'] = total_start_with_upper_case_words

            if total_words > 0:
                features['ratio_of_lower_case_words'] = total_lower_case_words / total_words
                features['ratio_of_upper_case_words'] = total_upper_case_words / total_words
                features['ratio_of_start_with_lower_case_words'] = total_start_with_lower_case_words / total_words
                features['ratio_of_start_with_upper_case_words'] = total_start_with_upper_case_words / total_words
            else:
                features['ratio_of_lower_case_words'] = -1.0
                features['ratio_of_upper_case_words'] = -1.0
                features['ratio_of_start_with_lower_case_words'] = -1.0
                features['ratio_of_start_with_upper_case_words'] = -1.0

            if total_lower_case_words > 0:
                features['ratio_of_lower_case_to_start_with_lower_case_words'] = total_start_with_lower_case_words / total_lower_case_words
            else:
                features['ratio_of_lower_case_to_start_with_lower_case_words'] = -1.0

            if total_upper_case_words > 0:
                features['ratio_of_upper_case_to_start_with_upper_case_words'] = total_start_with_upper_case_words / total_upper_case_words
            else:
                features['ratio_of_upper_case_to_start_with_upper_case_words'] = -1.0

            features['number_of_letters'] = total_letters
            features['number_of_total_lower_case_letters'] = total_lower_case_letters
            features['number_of_total_upper_case_letters'] = total_upper_case_letters

            if total_letters > 0:
                features['ratio_of_lower_case_letters'] = total_lower_case_letters / total_letters
                features['ratio_of_upper_case_letters'] = total_upper_case_letters / total_letters
            else:
                features['ratio_of_lower_case_letters'] = -1.0
                features['ratio_of_upper_case_letters'] = -1.0

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
    output = {
        'has_subject': True,
        'is_fwd': False,
        'is_re': False,
        'is_fw': False
    }

    if subject is not None:
        output['is_' + subject] = True
    else:
        output['has_subject'] = False

    return output


def generate_number_of_links(email):
    output = {'number_of_links': 0}
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
    output['is_mailing_list_by_subject'] = re.search(r'^\[[A-Za-z_\-]{2,}\]', email['subject']) is not None if email['subject'] is not None else False
    # TODO: implement function to detect email address, domain, username, and maybe name.
    # output['is_mailing_list_by_address'] =

    return output
