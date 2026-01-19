"""
Centralized Constants for the Multimodal Diagnosis Project.

This module contains all shared constants used across the project:
- Symptom normalization maps (TYPO_MAP, SYNONYM_MAP, PLURAL_MAP)
- Column exclusion lists (NON_SYMPTOM_COLS)
- Disease filtering lists (DISEASES_TO_EXCLUDE)
"""

# =============================================================================
# SYMPTOM NORMALIZATION CONSTANTS
# =============================================================================

# Common typos found in medical symptom data
TYPO_MAP = {
    # Spelling errors
    'vomitting': 'vomiting',
    'apetite': 'appetite',
    'neusea': 'nausea',
    'dizzy': 'dizziness',
    'weakeness': 'weakness',
    'stiffeness': 'stiffness',
    'numbess': 'numbness',
    'paleness': 'pallor',
    'tireness': 'tiredness',
    'slowhealing': 'slow healing',
    'thirsty': 'thirst',
    'fefver': 'fever',
    'burpin': 'burping',
    'itchness': 'itchiness',
    
    # Phrase typos
    'loss of consiousness': 'loss of consciousness',
    'loss of apetite': 'loss of appetite',
    'lack of apetite': 'loss of appetite',
    'nausea and vomitting': 'nausea and vomiting',
    'ringing in ears': 'ringing in ear',
    
    # Data merge artifacts
    'regurgitation.1': 'regurgitation',
}

# Plural to singular mappings for medical symptoms
PLURAL_MAP = {
    'headaches': 'headache',
    'rashes': 'rash',
    'nosebleeds': 'nosebleed',
    'seizures': 'seizure',
    'chills': 'chills',  # Exception: keep as plural (mass noun)
    'muscle aches': 'muscle ache',
    'body aches': 'body ache',
    'ringing in ears': 'ringing in ear',
    'swelling of feets': 'swelling of feet',
    'bloody stools': 'bloody stool',
    'irregular heartbeats': 'irregular heartbeat',
    'swelling of eyelids': 'swelling of eyelid',
    'numbness in arms': 'numbness in arm',
    'numbness in legs': 'numbness in leg',
    'swollen glands': 'swollen lymph nodes',
    'swollen lymph glands': 'swollen lymph nodes',
}

# Synonym mappings - map variants to canonical form
SYNONYM_MAP = {
    # Pain locations
    'belly pain': 'abdominal pain',
    'stomach pain': 'abdominal pain',
    'tummy pain': 'abdominal pain',
    'abdominal distention': 'abdominal pain',
    'abdominal cramps': 'abdominal pain',
    'stomach cramps': 'abdominal pain',
    
    # Fatigue variants
    'tiredness': 'fatigue',
    'extreme tiredness': 'fatigue',
    'lethargy': 'fatigue',
    'feeling tired': 'fatigue',
    'feeling weak': 'weakness',
    'extreme fatigue': 'fatigue',
    
    # Voice/throat
    'hoarseness': 'hoarse voice',
    
    # Weight changes
    'losing weight': 'weight loss',
    'unexplained weight loss': 'weight loss',
    'unintentional weight loss': 'weight loss',
    'recent weight loss': 'weight loss',
    'unexplained weight gain': 'weight gain',
    'failure to gain weight': 'poor weight gain',
    
    # Consciousness
    'fainting': 'loss of consciousness',
    
    # Bloating
    'stomach bloating': 'bloating',
    'feeling bloated': 'bloating',
    'belching': 'bloating',
    'swollen abdomen': 'swollen abdomen',
    
    # Appetite
    'loss of apetite': 'loss of appetite',
    'lack of apetite': 'loss of appetite',
    'lack of appetite': 'loss of appetite',
    'poor apetite': 'loss of appetite',
    'poor appetite': 'loss of appetite',
    'decreased appetite': 'loss of appetite',
    
    # Nausea/vomiting
    'nausea and vomitting': 'nausea and vomiting',
    'vomiting blood': 'vomiting blood',
    'throwing up': 'vomiting',
    
    # Swallowing
    'difficulty in swallowing': 'difficulty swallowing',
    'trouble swallowing': 'difficulty swallowing',
    'pain when swallowing': 'difficulty swallowing',
    
    # Speech
    'trouble speaking': 'difficulty speaking',
    'slurring words': 'slurred speech',
    'slow speech': 'slurred speech',
    
    # Vision
    'blurry vision': 'blurred vision',
    'blurred vision': 'blurred vision',
    'trouble with vision': 'vision problems',
    'vision less clear': 'vision loss',
    'light sensitivity': 'sensitivity to light',
    'redness of eye': 'eye redness',
    'eye inflammation': 'eye redness',
    'pink eye': 'eye redness',
    'involuntary eye movements': 'eye moves abnormally',
    
    # Skin
    'changes in skin color': 'skin color changes',
    'skin flushing': 'flushing',
    'itching': 'itching of skin',
    'itchy skin': 'itching of skin',
    'yellowing of skin': 'jaundice',
    'yellow skin': 'jaundice',
    'pale skin': 'pallor',
    
    # Sensation/Coordination
    'lack of coordination': 'loss of coordination',
    'poor coordination': 'loss of coordination',
    'loss of touch': 'loss of sensation',
    'reduced pain sensation': 'loss of sensation',
    'loss of balance': 'problems with balance',
    'dizzy': 'dizziness',
    
    # Heart
    'fast heart beat': 'fast heart rate',
    'fast heartbeats': 'fast heart rate',
    'pounding heart': 'heart palpitations',
    'heart failure': 'heart palpitations',
    
    # Swelling/Lumps
    'swelling of legs': 'leg swelling',
    'swollen legs': 'leg swelling',
    'swelling in legs': 'leg swelling',
    'swelling of feet': 'foot or toe swelling',
    'swollen feet': 'foot or toe swelling',
    'swollen toes': 'foot or toe swelling',
    'swelling of hands': 'hand or finger swelling',
    'swollen fingers': 'hand or finger swelling',
    'facial swelling': 'swelling',
    'swollen tonsils': 'swollen or red tonsils',
    'lump on vulva': 'mass on vulva',
    'swollen lymph glands': 'swollen lymph nodes',
    
    # Other
    'pain during sex': 'pain during intercourse',
    'severe headache': 'headache',
    'pain in joint': 'joint pain',
    'muscle tightness': 'muscle stiffness or tightness',
    'blood in feces': 'blood in stool',
    'bloody stool': 'blood in stool',
    'muscle or joint pain': 'muscle pain',
    'dry mouth and lips': 'dry mouth',
    'gas pain': 'gas',
    'loss of smell': 'disturbance of smell or taste',
    'discomfort in chest': 'chest pain',
    'heavy menstrual periods': 'heavy menstrual flow',
    'irregular periods': 'irregular menstrual cycles',
    'high bmi': 'obesity',
}


# =============================================================================
# DATA CLEANING CONSTANTS
# =============================================================================

# Columns that are NOT symptom features
NON_SYMPTOM_COLS = {
    'diseases', 'disease_category', 'symptoms', 
    'age', 'gender', 'sex', 'age_group', 
    'weight', 'height', 'bmi', 'occupation'
}

# Diseases to exclude from the dataset (diagnosed via other means, not symptoms)
DISEASES_TO_EXCLUDE = [
    # --- TRAUMATIC INJURIES AND ORTHOPEDICS ---
    # Rationale: Diagnosed via physical trauma history or X-ray/imaging
    'birth trauma', 'bone spur of the calcaneous', 'concussion', 'corneal abrasion', 
    'crushing injury', 'dislocation of the ankle', 'dislocation of the elbow', 
    'dislocation of the finger', 'dislocation of the foot', 'dislocation of the hip', 
    'dislocation of the knee', 'dislocation of the patella', 'dislocation of the shoulder', 
    'dislocation of the vertebra', 'dislocation of the wrist', 'fracture of the ankle', 
    'fracture of the arm', 'fracture of the facial bones', 'fracture of the finger', 
    'fracture of the foot', 'fracture of the hand', 'fracture of the jaw', 
    'fracture of the leg', 'fracture of the neck', 'fracture of the patella', 
    'fracture of the pelvis', 'fracture of the rib', 'fracture of the shoulder', 
    'fracture of the skull', 'fracture of the vertebra', 'head injury', 
    'heart contusion', 'hematoma', 'injury of the ankle', 'injury to internal organ', 
    'injury to the abdomen', 'injury to the arm', 'injury to the face', 
    'injury to the finger', 'injury to the hand', 'injury to the hip', 
    'injury to the knee', 'injury to the leg', 'injury to the shoulder', 
    'injury to the spinal cord', 'injury to the trunk', 'joint effusion', 
    'knee ligament or meniscus tear', 'lung contusion', 'rotator cuff injury', 
    'sprain or strain',

    # --- OPEN WOUNDS AND POST-SURGICAL ---
    # Rationale: Diagnosed by visual inspection; acute physical findings
    'infection of open wound', 'open wound due to trauma', 'open wound from surgical incision', 
    'open wound of the abdomen', 'open wound of the arm', 'open wound of the back', 
    'open wound of the cheek', 'open wound of the chest', 'open wound of the ear', 
    'open wound of the eye', 'open wound of the face', 'open wound of the finger', 
    'open wound of the foot', 'open wound of the head', 'open wound of the jaw', 
    'open wound of the knee', 'open wound of the lip', 'open wound of the mouth', 
    'open wound of the neck', 'open wound of the nose', 'open wound of the shoulder', 
    'pain after an operation', 'postoperative infection', 'burn',

    # --- MINOR AILMENTS AND ROUTINE ILLNESSES ---
    # Rationale: Low-stakes diagnoses identified in seconds
    'acne', 'actinic keratosis', "athlete's foot", 'broken tooth', 'bunion', 
    'callus', 'chalazion', 'chickenpox', 'cold sore', 'common cold', 
    'dental caries', 'diaper rash', 'ear wax impaction', 
    'flat feet', 'flu', 'gum disease', 'hammer toe', 'impetigo', 'ingrown toe nail', 
    'intertrigo (skin condition)', 'lice', 'mumps', 'oral thrush (yeast infection)', 
    'pinguecula', 'pityriasis rosea', 'scabies', 'sebaceous cyst', 
    'seborrheic keratosis', 'skin polyp', 'stye', 'teething syndrome', 
    'tooth abscess', 'tooth disorder', 'viral warts',

    # --- EXTERNAL CAUSES, POISONING, AND ENVIRONMENT ---
    # Rationale: Diagnosis based on toxicology or patient history of exposure
    'carbon monoxide poisoning', 'envenomation from spider or animal bite', 
    'frostbite', 'heat exhaustion', 'heat stroke', 'hypothermia', 
    'insect bite', 'insulin overdose', 'poisoning due to analgesics', 
    'poisoning due to anticonvulsants', 'poisoning due to antidepressants', 
    'poisoning due to antimicrobial drugs', 'poisoning due to antipsychotics', 
    'poisoning due to antihypertensives', 'poisoning due to ethylene glycol', 
    'poisoning due to gas', 'poisoning due to opioids', 'poisoning due to sedatives', 
    'drug poisoning due to medication', 'alcohol intoxication',

    # --- PHYSIOLOGICAL STATES AND LIFESTYLE ---
    # Rationale: Pregnancy/Menopause are life stages; substance abuse is behavioral history
    'pregnancy', 'menopause', 'fetal alcohol syndrome', 'normal pressure hydrocephalus', 
    'induced abortion', 'spontaneous abortion', 'missed abortion', 'mastectomy', 
    'alcohol abuse', 'drug abuse', 'drug abuse (barbiturates)', 'drug abuse (cocaine)', 
    'drug abuse (methamphetamine)', 'drug abuse (opioids)', 'marijuana abuse', 
    'smoking or tobacco addiction',

    # --- VISUAL FINDINGS ---
    # Rationale: Doctors can see these immediately
    'eye alignment disorder'
]
