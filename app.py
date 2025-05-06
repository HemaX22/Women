from flask import Flask, render_template, request, redirect, url_for, flash, abort
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models.database import db, init_db
from models.user import User
from models.complaint import Complaint
import os
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import string
import nltk

app = Flask(__name__)
app.config.from_pyfile('config.py')

# Initialize database
init_db(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class ComplaintClassifier:
    def __init__(self):
        if app.config.get('SKIP_NLP_INIT', False):
            self.vectorizer = None
            self.classifier = None
            return
            
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            self.vectorizer = TfidfVectorizer()
            self.classifier = MultinomialNB()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Initialize with proper training data
            self.train_data = self._load_training_data()
            self.train_model()
        except Exception as e:
            app.logger.error(f"Error initializing NLP: {e}")
            self.vectorizer = None
            self.classifier = None
    
    def _load_training_data(self):
        """Load and properly format training data"""
        data = {'text': [
            'manager keeps commenting on my body and outfits',
            'boss implied I should sleep with him for a promotion',
            'male coworkers make sexual jokes about female colleagues',
            'supervisor touches my waist when passing by',
            'excluded from important projects because Im a woman',
            'client made inappropriate advances during meeting',
            'HR dismissed my harassment complaint without investigation',
            'colleagues call me emotional when I voice opinions',
            'asked to wear more revealing clothes for client dinners',
            'senior executive sent flirtatious texts late at night',
            'team outing at strip club made mandatory',
            'promotion given to less qualified male colleague',
            'work events always scheduled at bars with heavy drinking',
            'called bossy when leading a team meeting',
            'jokes about women belonging in the kitchen during lunch',
            'husband slaps me during arguments',
            'partner monitors all my phone calls and messages',
            'mother-in-law threatens to take away my children',
            'spouse locks me out of the house as punishment',
            'forced into sexual acts multiple times weekly',
            'partner smashes my phone when angry',
            'family says I deserve the beatings for talking back',
            'husband installed spyware on my laptop',
            'not allowed to visit friends or family alone',
            'spouse withholds money for basic necessities',
            'strangled me during last nights fight',
            'threatens to kill me if I report to police',
            'destroyed my birth certificate and documents',
            'forces me to beg for grocery money',
            'punishes children when I disobey him',
            'professor asked for nude photos in exchange for grades',
            'stranger masturbated while staring at me on bus',
            'colleague keeps "accidentally" brushing against me',
            'neighbor takes photos of me through bathroom window',
            'received dick pics from supervisor on WhatsApp',
            'group of men groped me in crowded market',
            'club security demanded blowjob to let me in',
            'taxi driver locked doors and asked for kiss',
            'doctor made unnecessary breast examinations',
            'gym trainer touches me unnecessarily during sessions',
            'construction workers catcall me daily',
            'boss asked about my sexual fantasies',
            'delivery man won\'t leave until I give him my number',
            'colleague spread rumors about sleeping with me',
            'man followed me home whistling obscene songs',
            'denied director position despite 10 years experience',
            'male team members earn 40% more for same work',
            'told women cant handle technical roles in our company',
            'forced to take notes in meetings despite equal rank',
            'rejected from coding bootcamp for being female',
            'expected to organize office parties as only woman',
            'job ad said looking for young male candidates',
            'assumed I wouldnt relocate because of kids',
            'client refused to work with female engineer',
            'passed over for promotion due to pregnancy',
            'told to smile more during presentations',
            'performance review criticized me for being assertive',
            'company paid for male colleagues MBA but not mine',
            'asked if my husband approves of my work hours',
            'meetings scheduled during school pickup times'
            'parents arranging marriage for my 15 year old sister',
            'village panchayat ordered underage wedding',
            'brother goes to school while I do household work',
            'told marriage is only way out of poverty',
            'school principal ignored my child marriage complaint',
            'engaged at 14 to 35 year old man',
            'trafficked from Bihar to Punjab as child bride',
            'no police help when reported forced marriage',
            'married at 12, now have two children at 16',
            'parents took me out of school for marriage',
            'husband 20 years older beats me daily',
            'local NGO refused to help stop my marriage',
            'community says educated girls dont get husbands',
            'father sold me to pay gambling debts',
            'mother says suffering builds character',
            'ex-boyfriend posted intimate photos online',
            'anonymous accounts sending rape threats on Twitter',
            'colleague created fake dating profile with my photos',
            'online stalker knows my daily routine and addresses',
            'received 100+ obscene messages from unknown numbers',
            'group chat sharing my morphed nude images',
            'hacked my social media and posted vile content',
            'former friend leaking my private conversations',
            'dating app match turned into blackmailer',
            'boss sending inappropriate messages on LinkedIn',
            'men offering money for sexual favors on Instagram',
            'WhatsApp group rating local women\'s attractiveness',
            'deepfake porn videos circulating with my face',
            'threats to release videos unless I pay bitcoin',
            'fake reviews claiming I offer sexual services',
            'in-laws demanding luxury car as additional dowry',
            'husband beats me when my parents cant give money',
            'forced to sign property papers in husband\'s name',
            'called worthless for bringing insufficient dowry',
            'mother-in-law starves me as punishment',
            'threatened with divorce if more dowry not given',
            'jewelry gifted by parents taken by husband\'s family',
            'locked out of house until father pays 10 lakhs',
            'husband took second wife because of insufficient dowry',
            'in-laws publicly humiliate me about my family\'s status',
            'not allowed to visit parents until they pay more',
            'all my salary goes to husband\'s family',
            'forced to do all household work like servant',
            'medical treatment denied to pressure for money',
            'children kept from me as dowry leverage',
            'group of men followed me for 6 blocks',
            'eve-teasing at bus stop every morning',
            'men taking upskirt photos in crowded markets',
            'auto driver tried to kiss me during ride',
            'public masturbation near school gate',
            'men blocking path and demanding phone number',
            'street vendor showing porn and grinning',
            'parked car surrounded by men making gestures',
            'security guard watching me change in trial room',
            'men exposing themselves near college campus',
            'drunk men grabbing me during festivals',
            'taxi driver took wrong route to isolated area',
            'men rubbing against me in crowded trains',
            'followed home by strangers multiple times',
            'public place urination while staring at me',
            'police refused to file my domestic violence complaint',
            'court case delayed for 5 years with no progress',
            'lawyer asked for sexual favors to take my case',
            'property rights denied despite court order',
            'local politician interfering with rape case',
            'not allowed to enter temple during menstruation',
            'husband won divorce but wont pay alimony',
            'police sided with attacker because hes influential',
            'village council ordered rape victim to marry attacker',
            'court denied abortion for 14-year-old rape survivor',
            'employer fired me after sexual harassment complaint',
            'police mocked my cybercrime complaint',
            'landlord evicted me for being single woman',
            'hospital denied treatment without male relative',
            'passport office demanded husband\'s permission',
            'fired after announcing pregnancy',
            'not considered for promotion because might have kids',
            'only woman in team always given clerical work',
            'performance standards higher for female employees',
            'company policy denies period leave',
            'job reassigned after maternity leave return',
            'comments about age and marital status in interviews',
            'client meetings scheduled at late-night hotels',
            'no lactation room despite legal requirement',
            'travel restrictions for "safety concerns"',
            'mandatory high heels policy for female staff',
            'assigned less important projects than male peers',
            'performance review mentions "hormonal mood swings"',
            'team building at male-dominated sports events',
            'asked about family planning during promotion review',
            'Manager commented on my body during meeting',
            'Boss keeps asking me out for drinks after work',
            'Colleague showed me explicit content at work',
            'Supervisor brushed against me intentionally',
            'Excluded from project because I rejected advances',
            'HR dismissed my harassment complaint',
            'Client made sexual jokes about female staff',
            'Promotion denied after refusing date with boss',
            'Coworker keeps calling me pet names',
            'Asked to wear revealing clothes for client meeting',
            'Senior employee sending flirtatious texts',
            'Team outing turned into uncomfortable situation',
            'Male colleagues rating female employees',
            'Threatened with bad review if I complain',
            'Unwanted shoulder massages from supervisor',
            'Office gossip about my personal life',
            'Client touched me inappropriately at event',
            'Photoshopped images of me circulated at work',
            'Asked about my relationship status repeatedly',
            'Comments about my clothes being too tight',
            'Job offer contingent on personal favors',
            'Forced to sit through sexist presentations',
            'Manager watches me work from his office',
            'Colleagues making bets about who can date me',
            'HR suggested I take the comments as compliments',
            'Boss insists on private late-night meetings',
            'Coworker keeps "accidentally" touching me',
            'Client asked if I provide "extra services"',
            'Team leader shares inappropriate memes',
            'Comments about how marriage will affect my work',
            'Asked to join "men-only" golf outings',
            'Senior staff making remarks about my pregnancy',
            'Colleague keeps asking about my sex life',
            'Manager suggested I smile more to get ahead',
            'Received promotion offer with sexual conditions',
            'Coworker follows me to my car after work',
            'Client demanded kiss to sign contract',
            'Comments about my appearance during reviews',
            'Team leader insists on hotel room meetings',
            'HR said harassment is part of company culture',
            'Husband locked me in the house all day',
            'Partner destroyed my phone to isolate me',
            'Mother-in-law forces me to do all housework',
            'Spouse threatens to take away my children',
            'Forced to hand over my salary every month',
            'Husband monitors all my social media',
            'Partner prevents me from seeing friends',
            'Family supports his abuse saying its normal',
            'He hits me then buys gifts to apologize',
            'Not allowed to visit my parents',
            'Forced to quit job to stay at home',
            'Threatened with divorce if I speak up',
            'Husband controls what I wear outside',
            'Partner throws objects at me during fights',
            'Denied medical care after injuries',
            'Spouse threatens suicide if I leave',
            'Forced to have sex after physical abuse',
            'Husband hides my important documents',
            'Not allowed to learn driving',
            'Partner sabotages my birth control',
            'Financial abuse - no access to bank accounts',
            'Threatened to share private photos online',
            'Forced to cook at 3AM when he demands',
            'Husband threatens my family if I complain',
            'Not allowed to attend family functions',
            'Partner controls when I can use bathroom',
            'Spouse withholds money for basic needs',
            'Forced to sleep on floor as punishment',
            'Husband belittles me in front of guests',
            'Not allowed to make phone calls',
            'Partner forces me to beg for money',
            'Threatened with false police complaints',
            'Spouse damages things I care about',
            'Forced to fake happiness in public',
            'Husband tracks my location via GPS',
            'Not allowed to go to temple/mosque',
            'Partner forces me to wear what he chooses',
            'Threatened with honor killing',
            'Spouse abuses our pets to scare me',
            'Forced to tolerate his extramarital affairs',
            'Professor asked for favors in exchange for grades',
            'Stranger masturbated at me on bus',
            'Neighbor keeps peeping into my windows',
            'Doctor made inappropriate comments during exam',
            'Gym trainer touched me unnecessarily',
            'Colleague sent dick pics anonymously',
            'Taxi driver asked about my virginity',
            'Boss suggested threesome for promotion',
            'Client grabbed my waist during meeting',
            'Landlord demands sex for rent reduction',
            'Classmate spread rumors about my body',
            'Shopkeeper touched while handing change',
            'Relative forced kiss at family event',
            'Online date tried to remove condom secretly',
            'Photographer asked me to pose nude',
            'Delivery man made lewd gestures',
            'Barista wrote phone number on my cup',
            'Co-passenger rubbed against me in train',
            'Tailor measured too close to my breasts',
            'Driver took wrong route to isolated area',
            'Boss suggested I wear shorter skirts',
            'Stranger followed me for 3 blocks',
            'Doctor commented on my sexual history',
            'Neighbor left porn at my doorstep',
            'Colleague described sexual fantasies',
            'Client sent hotel room key with contract',
            'Trainer adjusted my posture unnecessarily',
            'Shop employee stared at my chest',
            'Classmate took upskirt photos',
            'Boss booked single hotel room for trip',
            'Driver locked car doors and smiled',
            'Doctor insisted on unnecessary exams',
            'Neighbor filmed me without consent',
            'Colleague "accidentally" sent sex tape',
            'Client suggested weekend getaway',
            'Gym member took photos in locker room',
            'Professor offered better grades for dates',
            'Stranger ejaculated on me in crowd',
            'Massage therapist touched genitals',
            'Boss commented on my sex appeal to clients',
            'Paid less than male colleague same role',
            'Rejected from engineering college for being female',
            'Told tech jobs are not for women',
            'Forced to make coffee for all male team',
            'Promotion given to less qualified man',
            'Job ad said "preferred male candidate"',
            'Excluded from important client meetings',
            'Comments about hormonal mood swings',
            'Assumed I would quit after marriage',
            'Not considered for leadership role',
            'Told to "act more feminine" at work',
            'Male interns given better projects',
            'Performance standards higher for me',
            'Asked if I plan to have children',
            'Male colleague took credit for my idea',
            'Called "too aggressive" for same behavior',
            'Not allowed to handle certain accounts',
            'Stereotyped as "emotional decision maker"',
            'Dress code stricter for women',
            'Meeting scheduled at men-only club',
            'Travel restrictions male colleagues dont have',
            'Assumed I wouldnt understand technical details',
            'Pregnancy treated as performance issue',
            'Called "diversity hire" behind my back',
            'Not invited to after-work networking',
            'Comments about "women belonging in kitchen"',
            'Work judged more harshly than male peers',
            'Asked to organize parties instead of projects',
            'Called "difficult" for asserting opinions',
            'Male colleague promoted despite my better results',
            'Assumed I needed help with technical tasks',
            'Told my voice is "too shrill" for leadership',
            'Not given company car like male counterparts',
            'Performance review focused on my appearance',
            'Important emails cc\'d only to male colleagues',
            'Called "bossy" for same behavior praised in men',
            'Assumed I got job through connections',
            'Meeting times changed to accommodate "family men"',
            'Ideas ignored until repeated by male colleague',
            'Told I "got lucky" with achievements',
            'Parents marrying me at 16 to pay debts',
            'Brother allowed school but I must marry',
            'Uncle forcing me to marry 40yr old man',
            'Village panchayat ordered underage marriage',
            'Parents hiding my birth certificate',
            'Engaged since age 12 to cousin',
            'School expelled me for being married',
            'Husband twice my age locks me at home',
            'Parents took dowry for my child marriage',
            'Forced to quit school for marriage',
            'Threatened with honor killing if I refuse',
            'Marriage certificate shows fake age',
            'Trafficked to another state as child bride',
            'Husband abuses me but parents say stay',
            'Pregnant at 15 from forced marriage',
            'No ID proof to report underage marriage',
            'Community celebrates child marriages',
            'Local leader involved in trafficking',
            'Police refused to file my complaint',
            'Teachers ignored my marriage plans',
            'Ex posted intimate photos online',
            'Fake profile with my photos soliciting sex',
            'Colleague created deepfake porn of me',
            'Online stalker sending 100+ messages daily',
            'Blackmailed with private chats',
            'Group chat rating female classmates',
            'Morphed images circulated in college',
            'Dating app match turned violent threats',
            'Work email receiving porn links',
            'Social media comments about my body',
            'Hacked accounts to find private info',
            'Anonymous blog spreading false rumors',
            'Photoshopped nude images sent to employer',
            'Twitch streamer doxxing female gamers',
            'Revenge porn shared in WhatsApp groups',
            'Fake matrimonial profile with my number',
            'AI-generated voice clips used for blackmail',
            'Gaming community sending rape threats',
            'LinkedIn messages with sexual demands',
            'Zoom bombing with explicit content',
            'Senior asked me to sit on his lap during appraisal',
            'Client said he only works with "pretty female reps"',
            'Team lead shares my personal number without consent',
            'Colleagues betting on who can "score" with me first',
            'HR asked what I wore during harassment incident',
            'Boss calls me into office just to stare at me',
            'Forced to attend late-night "work" dinners alone',
            'Manager said my job depends on being "friendly"',
            'Coworker keeps "accidentally" sending nudes',
            'Promised raise if I go on weekend trip with boss',
            'Office security guard watches me change clothes',
            'Colleague installed spy cam in women\'s restroom',
            'Told to tolerate harassment for "team harmony"',
            'Client demands I drink alcohol to sign deal',
            'HR manager asked for nude photos "for my file"',
            'Boss comments on how pregnancy changed my body',
            'Team outing at strip club made mandatory',
            'Colleague spread rumor I slept my way to promotion',
            'Daily messages asking if I\'m single/married',
            'Job interview in hotel room instead of office',
            'Contract includes "physical availability" clause',
            'Fired after refusing to massage my supervisor',
            'Coworker keeps sniffing my hair/handkerchief',
            'Forced to wear heels despite medical condition',
            'HR said harassment is "just office flirting"',
            'Boss gifts lingerie as "performance incentive"',
            'Colleague keeps proposing in front of everyone',
            'Meeting notes include comments about my body',
            'Security checks only on female employees',
            'Mandatory "body measurements" for uniform',
            'Client insists on only female staff for projects',
            'Office party games with sexual undertones',
            'Colleague hacked my social media to stalk me',
            'Manager said I "owe him" for hiring me',
            'Coworker keeps putting drugs in my coffee',
            'HR requires harassment victims to sign NDAs',
            'Boss makes me model office clothes for him',
            'Colleague follows me home "for protection"',
            'Annual review includes "sex appeal" rating',
            'Forced to share hotel room on business trips',
            'Office culture of rating female new hires',
            'Manager keeps "joking" about kidnapping me',
            'Coworker created fake dating profile of me',
            'Promotion depends on "personal loyalty"',
            'Security guards take upskirt photos',
            'Client gifts include sex toys and lingerie',
            'HR said I should be "flattered" by attention',
            'Colleague keeps stealing my undergarments',
            'Mandatory "wife approval" for male staff only',
            'Boss threatens to share nudes if I quit',
            'Husband forces me to have abortions',
            'Partner starves me as punishment',
            'Mother-in-law poisons my food regularly',
            'Spouse locks me out of house at night',
            'Forced to sign false confessions',
            'Husband gives me sleeping pills daily',
            'Partner burns me with cigarettes',
            'Family performs exorcisms on me',
            'Spouse forces me to beg on streets',
            'Husband shaves my head as humiliation',
            'In-laws force me to do manual labor',
            'Partner isolates me from all relatives',
            'Spouse makes me sleep with animals',
            'Husband tattoos his name on my body',
            'Forced to undergo virginity tests',
            'Partner chokes me unconscious regularly',
            'Mother-in-law steals my jewelry',
            'Husband sold my wedding gifts',
            'Spouse controls when I can bathe',
            'Forced to wear torn clothes at home',
            'Partner records my screams for fun',
            'Husband denies me medical treatment',
            'Family forces me to fast for husband',
            'Spouse makes me bark like a dog',
            'Husband forces me to watch porn',
            'Partner cuts me with kitchen knives',
            'Mother-in-law spits in my food',
            'Spouse ties me up for hours',
            'Husband forces me to drink alcohol',
            'Forced to crawl instead of walk',
            'Partner urinates on my belongings',
            'Husband makes me eat from floor',
            'Family performs black magic on me',
            'Spouse breaks bones as punishment',
            'Forced to breastfeed adult husband',
            'Partner burns my childhood photos',
            'Husband sells my clothes for money',
            'Mother-in-law forces me to miscarry',
            'Spouse injects me with unknown drugs',
            'Husband makes me beg for food',
            'Partner forces me to watch him cheat',
            'Family locks me in dark room for days',
            'Spouse brands me with hot iron',
            'Husband forces me to fake smiles',
            'Forced to call husband "master"',
            'Partner makes me lick his feet',
            'Mother-in-law beats me with broom',
            'Spouse shames me about past trauma',
            'Husband forbids me from crying',
            'Doctor filmed me during examination',
            'Yoga teacher touches in "adjustments"',
            'Auto driver showed me porn on phone',
            'College fest organizers groped girls',
            'Landlord enters my room while sleeping',
            'Tailer insists on measuring bare body',
            'Gym trainer "accidentally" removes bra',
            'Bus conductor presses against me',
            'Religious guru demanded sexual favors',
            'Police officer asked for bribe in sex',
            'College canteen worker spies in washroom',
            'Shopkeeper cameras in changing room',
            'Hospital staff watches me change',
            'Plumber touched me while "working"',
            'Electrician took photos under my skirt',
            'Wedding photographer demanded nudes',
            'Swimming coach touches in water',
            'Massage therapist exposes himself',
            'Driver showed me his genitals',
            'Neighbor masturbates while watching me',
            'Building watchman collects my underwear',
            'Delivery boy asks for kiss for package',
            'Cable guy installed camera in bedroom',
            'Priest touches inappropriately during puja',
            'Dance teacher forces close contact',
            'Hotel staff enters bathroom unannounced',
            'Park attendant blocks my path',
            'Street vendor grabs my hand forcefully',
            'College lab assistant locks me inside',
            'Library staff follows me between racks',
            'Coaching teacher sends dirty messages',
            'Sports coach selects only "pretty" girls',
            'Hostel warden checks rooms at night',
            'Exhibition staff takes upskirt photos',
            'Beautician films during waxing',
            'Spa employee adds hidden camera',
            'Swimming pool guard stares at girls',
            'PG owner enters rooms without knock',
            'College bus driver makes vulgar comments',
            'Canteen worker spikes girls\' food',
            'Temple priest touches during blessings',
            'Hospital compounder watches patients',
            'Parking attendant blocks car doors',
            'Tour guide forces jungle "massage"',
            'Photographer insists on nude shots',
            'Fitness trainer shares client photos',
            'Shop owner gropes during fittings',
            'Bus passenger rubs against standing girls',
            'Train TTE demands sex for ticket check',
            'Airport security pats down unnecessarily',
            'All female staff must clean office',
            'Male employees get better chairs',
            'Women not allowed in factory areas',
            'Separate lower standards for women',
            'No women allowed in board meetings',
            'Female employees must serve food',
            'Mandatory makeup for customer-facing roles',
            'No night shifts for women "for safety"',
            'Period tracking for female staff',
            'Women excluded from technical training',
            'Motherhood treated as performance issue',
            'Forced to resign after miscarriage',
            'No women in leadership pipeline',
            'Salary deductions for "emotional days"',
            'Mandatory pregnancy tests for hiring',
            'No travel allowance for female staff',
            'Women not given company laptops',
            'Separate (worse) insurance for women',
            'No parking spaces for female employees',
            'Female interns only get admin work',
            'Period leave requires doctor note',
            'No women in company sports teams',
            'Female engineers given only documentation',
            'Saleswomen must hug clients "for deals"',
            'No women in R&D department',
            'Separate (longer) probation for women',
            'Female staff must clean after parties',
            'No women in company\'s core committee',
            'Forced to sign "no pregnancy" contract',
            'Male staff get first pick of shifts',
            'Women not allowed to speak in meetings',
            'Female candidates asked about marriage',
            'No women in field operations team',
            'Separate (stricter) dress code for women',
            'Female employees must organize festivals',
            'No women in company strategy sessions',
            'Maternity leave counted as absenteeism',
            'Women not given access to tools',
            'Female staff must decorate office',
            'No women in technical interviews',
            'Separate (lower) food budget for women',
            'Female engineers only allowed testing',
            'No women in company hackathons',
            'Forced to take pay cut after marriage',
            'Women excluded from client entertainment',
            'No female representation in HR policies',
            'Separate (later) lunch breaks for women',
            'Female coders only allowed frontend',
            'No women in company decision-making',
            'Forced to quit after reporting harassment',
            'Parents sold me for marriage at 14',
            'Brother\'s education funded by my dowry',
            'Village head ordered my underage wedding',
            'Married at 15 to settle family dispute',
            'Husband 30 years older locks me inside',
            'Parents faked my age on marriage papers',
            'School expelled me when marriage discovered',
            'Local priest blessed child marriage',
            'Trafficked as bride to another country',
            'Pregnant at 14 from forced marriage',
            'Husband beats me for not doing housework',
            'Parents took loan against my marriage',
            'Married to pay father\'s medical bills',
            'Community celebrates child brides',
            'No birth certificate to prove my age',
            'Police refused to stop my wedding',
            'Teachers ignored signs of my marriage',
            'Husband controls when I can see family',
            'Forced to drop out in 8th standard',
            'Married to cousin against my will',
            'Parents hid marriage from authorities',
            'Husband\'s family controls my money',
            'No access to phone to call for help',
            'Threatened with acid attack if I escape',
            'Local leaders support child marriage',
            'Doctors ignored signs of abuse',
            'Husband forces me to work as maid',
            'Parents said marriage is my "destiny"',
            'No ID proof to access help services',
            'Community shames families who refuse',
            'Ex created fake porn videos of me',
            'Online stalker found my home address',
            'Colleague shared private chats at work',
            'Blackmailed with edited nude photos',
            'Fake matrimonial profile with my photos',
            'Hacker accessed my intimate videos',
            'Group chat sharing my morphed images',
            'Anonymous blog posting false rumors',
            'Dating app match turned blackmailer',
            'Work email receiving death threats',
            'Social media impersonation account',
            'Deepfake videos circulating in college',
            'Hacked cloud storage for private photos',
            'WhatsApp group rating female students',
            'Online gaming rape threats',
            'Employer checking deleted social media',
            'AI voice clone used for extortion',
            'Twitch streamer sharing my nudes',
            'Fake LinkedIn profile damaging reputation',
            'Zoom calls hacked with porn content',
            'Anonymous messages threatening rape',
            'Private photos leaked on torrent sites',
            'Fake COVID positive post with my number',
            'Doxxed for refusing online advances',
            'Tinder date stalking all my accounts',
            'Edited audio clips sent to my family',
            'Company forum posting my bra color',
            'Fake news about me being escort',
            'Instagram polls about my body',
            'Reddit thread discussing my private life',
            'In-laws demand luxury car as dowry',
            'Husband beats me for insufficient dowry',
            'Family pressures for more dowry after marriage',
            'Threatened with divorce for dowry',
            'Forced to bring money from parents',
            'In-laws list demands at wedding',
            'Denied food for not bringing dowry',
            'Husband compares me to richer wives',
            'Mother-in-law counts gifts publicly',
            'Threatened to marry richer girl',
            'Forced to sign dowry agreements',
            'Jewelry taken away after marriage',
            'In-laws harass for cash monthly',
            'Dowry demands increased after pregnancy',
            'Husband withholds money for dowry',
            'Family shames my parents for poverty',
            'Forced to work to pay dowry demands',
            'In-laws sell my wedding gifts',
            'Threatened with false police case',
            'Denied medical care over dowry',
            'Husband locks me out for more dowry',
            'In-laws control my salary for dowry',
            'Forced to abort female fetus for dowry',
            'Public humiliation for dowry demands',
            'Husband takes second wife for dowry',
            'In-laws force me to beg for dowry',
            'Threatened to kill for more dowry',
            'Forced to sign property papers',
            'Dowry negotiations at police station',
            'Husband files false cases for dowry',
            'Men follow me while jogging',
            'Group masturbating at me in park',
            'Bus driver won\'t stop near my house',
            'Street vendors block my path',
            'Men take photos without consent',
            'Public transport seat harassers',
            'Stalkers note my daily routine',
            'Men expose themselves in public',
            'Eve-teasing at market daily',
            'Auto drivers demand extra fare',
            'Men follow me home from college',
            'Public urinators target women',
            'Park benches occupied by harassers',
            'Men stare aggressively in metro',
            'Street food vendors touch unnecessarily',
            'Public transport groping gangs',
            'Men follow me into restrooms',
            'Stalkers wait outside workplace',
            'Men spit near me intentionally',
            'Public masturbators in theaters',
            'Men "accidentally" brush against',
            'Street photographers chase me',
            'Men block pathways to harass',
            'Public transport seat stealers',
            'Men record videos under skirts',
            'Street gamblers pass comments',
            'Men occupy women-only spaces',
            'Public drinking groups harass',
            'Men follow me in shopping malls',
            'Street hawkers grab my arms'
            ],
                'category': [
                'Workplace Harassment',
                'Workplace Harassment',
                'Workplace Harassment',
                'Workplace Harassment',
                'Gender Discrimination',
                'Workplace Harassment',
                'Workplace Harassment',
                'Gender Discrimination',
                'Workplace Harassment',
                'Workplace Harassment',
                'Workplace Harassment',
                'Gender Discrimination',
                'Workplace Harassment',
                'Gender Discrimination',
                'Gender Discrimination',
                'Domestic Violence',
                'Domestic Violence',
                'Domestic Violence',
                'Domestic Violence',
                'Domestic Violence',
                'Domestic Violence',
                'Domestic Violence',
                'Domestic Violence',
                'Domestic Violence',
                'Domestic Violence',
                'Domestic Violence',
                'Domestic Violence',
                'Domestic Violence',
                'Domestic Violence',
                'Domestic Violence',
                'Sexual Harassment',
                'Public Harassment',
                'Sexual Harassment',
                'Public Harassment',
                'Sexual Harassment',
                'Public Harassment',
                'Sexual Harassment',
                'Public Harassment',
                'Sexual Harassment',
                'Sexual Harassment',
                'Public Harassment',
    'Workplace Harassment',
    'Public Harassment',
    'Workplace Harassment',
    'Public Harassment',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Child Marriage',
    'Child Marriage',
    'Gender Discrimination',
    'Child Marriage',
    'Child Marriage',
    'Child Marriage',
    'Child Marriage',
    'Child Marriage',
    'Child Marriage',
    'Child Marriage',
    'Domestic Violence',
    'Child Marriage',
    'Gender Discrimination',
    'Domestic Violence',
    'Gender Discrimination',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Legal System Failure',
    'Legal System Failure',
    'Legal System Failure',
    'Legal System Failure',
    'Legal System Failure',
    'Cultural Discrimination',
    'Legal System Failure',
    'Legal System Failure',
    'Legal System Failure',
    'Legal System Failure',
    'Workplace Harassment',
    'Legal System Failure',
    'Gender Discrimination',
    'Cultural Discrimination',
    'Legal System Failure',
    'Workplace Harassment',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
        'Public Harassment',
    'Workplace Harassment',
    'Public Harassment',
    'Workplace Harassment',
    'Public Harassment',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Child Marriage',
    'Child Marriage',
    'Gender Discrimination',
    'Child Marriage',
    'Child Marriage',
    'Child Marriage',
    'Child Marriage',
    'Child Marriage',
    'Child Marriage',
    'Child Marriage',
    'Child Marriage',
    'Domestic Violence',
    'Child Marriage',
    'Gender Discrimination',
    'Domestic Violence',
    'Gender Discrimination',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Cyber Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Dowry Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Public Harassment',
    'Legal System Failure',
    'Legal System Failure',
    'Legal System Failure',
    'Legal System Failure',
    'Legal System Failure',
    'Cultural Discrimination',
    'Legal System Failure',
    'Legal System Failure',
    'Legal System Failure',
    'Legal System Failure',
    'Workplace Harassment',
    'Legal System Failure',
    'Gender Discrimination',
    'Cultural Discrimination',
    'Legal System Failure',
    'Workplace Harassment',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    'Gender Discrimination',
    
    
                ]

        }
        
        # Convert to DataFrame and ensure proper types
        df = pd.DataFrame(data)
        df['text'] = df['text'].astype(str)
        df['category'] = df['category'].astype(str)
        return df
    
    def train_model(self):
        """Train the model with proper data formatting"""
        if not hasattr(self, 'train_data') or self.train_data.empty:
            app.logger.error("No training data available")
            return
            
        try:
            self.train_data['processed_text'] = self.train_data['text'].apply(self.preprocess_text)
            X = self.vectorizer.fit_transform(self.train_data['processed_text'])
            y = self.train_data['category']
            
            # Ensure shapes match
            if X.shape[0] != len(y):
                raise ValueError("Feature and label dimensions don't match")
                
            self.classifier.fit(X, y)
            app.logger.info("Model trained successfully")
        except Exception as e:
            app.logger.error(f"Training failed: {e}")
            raise
    def predict_category(self, text):
        if not self.vectorizer or not self.classifier:
            return 'General Complaint'  # Fallback category
            
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        return self.classifier.predict(X)[0]
    
    def get_department(self, category):
        department_map = {
            # Original categories
            'Workplace Harassment': 'Internal Complaints Committee',
            'Domestic Violence': 'Women Protection Cell',
            'Child Marriage': 'Child Welfare Committee',
            'Gender Discrimination': 'State Women Commission',
            'Sexual Harassment': 'Special Women\'s Police Unit',
            'Workplace Discrimination': 'Labor Department',
            
            # New categories from expanded dataset
            'Cyber Harassment': 'Cyber Crime Cell',
            'Dowry Harassment': 'Dowry Prohibition Cell',
            'Public Harassment': 'Local Police Public Safety Wing',
            
            # Fallback for uncategorized complaints
            'General Complaint': 'District Legal Services Authority'
        }
        return department_map.get(category, 'District Legal Services Authority')

# Initialize classifier
classifier = ComplaintClassifier()

# ======================
# Authentication Routes
# ======================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('register'))
        
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful. Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember = True if request.form.get('remember') else False
        
        user = User.query.filter_by(username=username).first()
        
        if not user or not user.check_password(password):
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
        
        login_user(user, remember=remember)
        
        # Redirect to appropriate dashboard
        if user.is_admin:
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# ======================
# User Routes
# ======================

@app.route('/dashboard')
@login_required
def dashboard():
    user_complaints = Complaint.query.filter_by(user_id=current_user.id).order_by(Complaint.created_at.desc()).all()
    return render_template('dashboard.html', complaints=user_complaints)

@app.route('/file-complaint', methods=['GET', 'POST'])
@login_required
def file_complaint():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        is_anonymous = True if request.form.get('anonymous') else False
        
        # Classify complaint with fallback
        try:
            category = classifier.predict_category(f"{title} {description}")
            department = classifier.get_department(category)
        except:
            category = 'General Complaint'
            department = 'General Help Desk'
        
        complaint = Complaint(
            user_id=current_user.id,
            title=title,
            description=description,
            category=category,
            department=department,
            is_anonymous=is_anonymous
        )
        
        db.session.add(complaint)
        db.session.commit()
        
        flash('Your complaint has been submitted successfully!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('complaint.html')

@app.route('/complaint-status/<int:complaint_id>')
@login_required
def complaint_status(complaint_id):
    complaint = Complaint.query.get_or_404(complaint_id)
    if complaint.user_id != current_user.id and not current_user.is_admin:
        flash('You are not authorized to view this complaint', 'error')
        return redirect(url_for('dashboard'))
    
    return render_template('status.html', complaint=complaint)

# ======================
# Admin Routes
# ======================

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        abort(403)
    
    complaints = db.session.query(
        Complaint,
        User.username
    ).outerjoin(
        User, Complaint.user_id == User.id
    ).order_by(
        Complaint.created_at.desc()
    ).all()
    
    return render_template('admin/dashboard.html', complaints=complaints)

@app.route('/admin/update-status/<int:complaint_id>', methods=['POST'])
@login_required
def update_status(complaint_id):
    if not current_user.is_admin:
        abort(403)
    
    complaint = Complaint.query.get_or_404(complaint_id)
    new_status = request.form.get('status')
    
    if new_status not in ['Submitted', 'In Progress', 'Resolved', 'Rejected']:
        flash('Invalid status', 'error')
    else:
        complaint.status = new_status
        complaint.updated_at = datetime.utcnow()
        db.session.commit()
        flash('Status updated successfully', 'success')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete-complaint/<int:complaint_id>', methods=['POST'])
@login_required
def delete_complaint(complaint_id):
    if not current_user.is_admin:
        abort(403)
    
    complaint = Complaint.query.get_or_404(complaint_id)
    db.session.delete(complaint)
    db.session.commit()
    flash('Complaint deleted successfully', 'success')
    return redirect(url_for('admin_dashboard'))

if __name__ == '__main__':
    app.run(debug=True)