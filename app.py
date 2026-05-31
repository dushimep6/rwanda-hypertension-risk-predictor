# =============================================================================
# app.py — Hypertension Risk Prediction Tool
# Rwanda MOH — WHO STEPS Survey Nationwide
# CatBoost v2 · AUC=0.734 · 16 Monotonic Constraints
# =============================================================================
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd, joblib
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Hypertension Risk Prediction Tool — Rwanda",
    page_icon="❤️", layout="centered",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');
:root{
  --navy:#0B2545;--teal:#1A5276;--teal2:#1F618D;
  --green:#1A7A3C;--green-lt:#EAF7EF;--green-dk:#145A2E;
  --yellow:#B7770D;--yellow-lt:#FEF9E7;--yellow-bd:#F0D060;
  --orange:#B94A00;--orange-lt:#FEF0E7;--orange-bd:#E8874A;
  --red:#922B21;--red-lt:#FDEDEC;--red-bd:#E57373;
  --border:#D5DCE8;--text:#0B1E30;--text2:#3D5166;--text3:#7A8FA6;
  --bg:#F4F6FA;--card:white;--shadow:0 2px 12px rgba(11,37,69,0.08)
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;background:var(--bg)!important;color:var(--text)!important}
.main .block-container{padding-top:0!important;padding-bottom:3rem;max-width:780px}
#MainMenu,footer,header,.stDeployButton{visibility:hidden;display:none}

/* Header — Rwanda flag: Blue title | Yellow strip | Green language */
.hdr-blue{
  background:#00A1DE;
  padding:20px 28px;text-align:center}
.hdr-title{font-family:'DM Serif Display',serif;font-size:24px;font-weight:400;
           color:white;letter-spacing:0.3px;line-height:1.2;margin:0;
           text-shadow:0 1px 4px rgba(0,0,0,0.15)}
.hdr-yellow{background:#FAD201;height:12px}
.hdr-green{background:#20603D;padding:8px 0;text-align:center}
.lang-row{display:flex;justify-content:center;gap:10px}
.lbtn{background:transparent!important;border:1.5px solid rgba(255,255,255,0.6)!important;
      color:white!important;padding:5px 22px;border-radius:4px;
      font-size:12px;font-weight:600;letter-spacing:0.5px;
      text-decoration:none!important;cursor:pointer;display:inline-block}
.lbtn:hover{background:rgba(255,255,255,0.15)!important;color:white!important;
            text-decoration:none!important}
.lbtn:visited{color:white!important}
.lbtn:link{color:white!important}
.lbtn.on{background:rgba(255,255,255,0.22)!important;
         border-color:white!important;font-weight:700;color:white!important}

/* Section labels */
.slbl{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
      color:var(--teal2);margin:18px 0 10px;padding-bottom:5px;
      border-bottom:1px solid var(--border)}

/* Card */
.card{background:var(--card);border-radius:8px;border:1px solid var(--border);
      padding:16px 20px;margin:8px 0;box-shadow:var(--shadow)}

/* Inputs */
label,div[data-testid="stRadio"] label,div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label{
  font-family:'DM Sans',sans-serif!important;font-weight:600!important;
  font-size:11px!important;color:var(--text2)!important;
  text-transform:uppercase!important;letter-spacing:0.8px!important}
div[data-testid="stNumberInput"] input{
  font-family:'DM Sans',sans-serif!important;font-size:20px!important;
  font-weight:700!important;color:var(--navy)!important;text-align:center!important;
  border:2px solid var(--border)!important;border-radius:6px!important}

/* Button */
.stButton>button{
  background:var(--navy)!important;color:white!important;
  border:none!important;border-radius:6px!important;
  font-family:'DM Sans',sans-serif!important;font-weight:700!important;
  font-size:13px!important;letter-spacing:1.5px!important;
  text-transform:uppercase!important;padding:14px 20px!important;
  width:100%!important;margin-top:6px!important;
  box-shadow:0 4px 14px rgba(11,37,69,0.25)!important}
.stButton>button:hover{background:var(--teal)!important}

/* Reset button — same height as Calculate Risk, outlined */
div[data-testid="column"]:last-child .stButton>button{
  background:white!important;color:var(--navy)!important;
  border:2px solid var(--navy)!important;
  font-size:13px!important;font-weight:700!important;
  padding:14px 10px!important;letter-spacing:1.5px!important;
  box-shadow:none!important;width:100%!important;
  text-transform:uppercase!important;
  margin-top:0!important}
div[data-testid="column"]:last-child .stButton>button:hover{
  background:var(--navy)!important;color:white!important}

/* Alert badges */
.abadge{display:inline-block;padding:2px 8px;border-radius:3px;font-size:11px;
        font-weight:700;margin-left:6px;vertical-align:middle}
.abadge-warn{background:#FFF3CD;color:#856404;border:1px solid #FFEAA7}
.abadge-alert{background:#FFE0D0;color:#7D3300;border:1px solid #FFB899}
.abadge-ok{background:#D4EDDA;color:#155724;border:1px solid #B8DFC9}

/* Alert row */
.alert-item{display:flex;align-items:flex-start;gap:10px;
            padding:8px 0;border-bottom:1px solid #F0F4F8}
.alert-item:last-child{border-bottom:none}
.alert-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;margin-top:5px}
.alert-name{font-size:13px;font-weight:700;color:var(--navy);margin-bottom:1px}
.alert-val{font-size:12px;color:var(--text2)}

/* Risk result */
.risk-banner{border-radius:8px;padding:20px 24px;margin:10px 0;border-left:5px solid}
.risk-low   {background:var(--green-lt); border-color:var(--green)}
.risk-mod   {background:var(--yellow-lt);border-color:var(--yellow)}
.risk-high  {background:var(--orange-lt);border-color:var(--orange)}
.risk-vhi   {background:var(--red-lt);   border-color:var(--red)}
.risk-pct   {font-family:'DM Serif Display',serif;font-size:42px;line-height:1;margin-bottom:4px}
.risk-label {font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px}
.risk-msg   {font-size:13px;line-height:1.7;margin-top:8px;padding-top:8px;border-top:1px solid rgba(0,0,0,0.08)}
.risk-meta  {font-size:10px;color:var(--text3);margin-top:4px}

/* Progress bar for risk */
.rbar-bg{background:#E8ECF2;border-radius:4px;height:8px;margin:10px 0}
.rbar   {height:100%;border-radius:4px;transition:width 0.6s ease}

/* Quality dots */
.qdots{display:flex;gap:5px;margin:8px 0 2px}
.qdot{height:4px;flex:1;border-radius:2px;background:var(--border);max-width:80px}
.qdot.on{background:var(--navy)}
.qlbl{font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:var(--text3)}

/* Contribution section */
.contrib-row{display:flex;justify-content:space-between;align-items:center;
             padding:7px 0;border-bottom:1px solid #F0F4F8}
.contrib-row:last-child{border-bottom:none}
.contrib-name{font-size:12px;font-weight:600;color:var(--navy)}
.contrib-dir {font-size:11px;font-weight:700}
.contrib-bar-bg{background:#EEF2F8;border-radius:2px;height:4px;width:80px}
.contrib-bar  {height:100%;border-radius:2px}

/* Recommendation */
.rec-item{padding:10px 0;border-bottom:1px solid #F0F4F8}
.rec-item:last-child{border-bottom:none}
.rec-num{display:inline-flex;align-items:center;justify-content:center;
         width:20px;height:20px;border-radius:50%;font-size:11px;
         font-weight:700;color:white;margin-right:8px;flex-shrink:0}
.rec-title{font-size:13px;font-weight:700;color:var(--navy);margin-bottom:2px}
.rec-desc{font-size:12px;color:var(--text2);line-height:1.6;margin-left:28px}

/* Disclaimer */
.disc{font-size:11px;color:var(--text3);text-align:center;
      padding:12px;border-top:1px solid var(--border);margin-top:16px;line-height:1.6}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:var(--card)!important;
  border-radius:6px!important;padding:4px!important;gap:2px!important;
  border:1px solid var(--border)!important;box-shadow:var(--shadow)!important;
  justify-content:center!important;width:fit-content!important;
  margin:0 auto!important}
.stTabs [data-baseweb="tab"]{border-radius:4px!important;
  font-family:'DM Sans',sans-serif!important;font-weight:600!important;
  font-size:11px!important;padding:8px 28px!important;color:var(--text2)!important;
  letter-spacing:0.5px!important;text-transform:uppercase!important}
.stTabs [aria-selected="true"]{background:var(--navy)!important;
  color:white!important;box-shadow:0 2px 8px rgba(11,37,69,0.2)!important}
</style>
""", unsafe_allow_html=True)

# ── Translations ──────────────────────────────────────────────────────────────
TX = {
'en':{
  'title':'Hypertension Risk Prediction Tool',
  'tab1':'ABOUT YOU','tab2':'ADD MEASUREMENTS',
  'l1':'Please answer the questions below to get your personal hypertension risk estimate.',
  'l2':'Add your body measurements to get a more accurate result.',
  'add_l1':'Please complete the About You tab first.',
  'about':'About You','life':'Your Lifestyle',
  'body':'Body Measurements',
  'age':'How old are you?','sex':'Sex','prov':'Province',
  'urb':'Where do you live?','educ':'Highest level of education completed',
  'pi':'How physically active are you?',
  'rur':'Rural','urbn':'Urban',
  'e1':'No schooling or Primary','e2':'Secondary','e3':'University or higher',
  'alc':'Do you drink alcohol?',
  'd5':'How often do you add salt to your food?',
  'd5_help':'Salt added at the table or during cooking',
  'd7':'How often do you eat salty or processed food?',
  'd7_help':'Examples: chips, crisps, tinned food, sausages, instant noodles',
  'diab':'Have you ever been told by a doctor that you have diabetes?',
  'diab_yes_badge':'Diabetes raises blood pressure risk',
  'wt':'Weight (kg)','ht':'Height (cm)',
  'wst':'Waist circumference (cm)',
  'wst_check':'I have measured my waist circumference',
  'wst_help':'Measure around your belly button with a tape measure',
  'wst_skip':'Waist not measured — will be estimated automatically',
  'recalc_note':'If you change any answer above, click Calculate My Risk again to update your result.',
  'btn1':'Calculate My Risk',
  'btn2':'Update My Risk',
  'reset':'Start Again',
  'sel_model':'Select Model',
  'bmi_normal':'Healthy weight','bmi_over':'Overweight','bmi_obese':'Obese',
  'wst_normal':'Normal','wst_high':'Above safe limit',
  'pa':'Active',
  'pa_badge':'At least 30 min on 5 days/week',
  'pi_':'Not active enough',
  'pi_badge':'Below 30 min on 5 days/week',
  'a0':'Never',
  'a1':'Within 12 months',
  'a2':'Past 30 days',
  's1':'Never',
  's2':'Rarely  (1–3 days/month)',
  's3':'Sometimes  (2–3 days/week)',
  's4':'Often  (4–5 days/week)',
  's5':'Always  (every day)',
  'pe':'Eastern Province','pk':'Kigali City',
  'pn':'Northern Province','ps':'Southern Province','pw':'Western Province',
  'fem':'Female','mal':'Male',
  'no':'No','yes':'Yes',
  'low':'Low Risk','mod':'Moderate Risk','high':'High Risk','vhi':'Very High Risk',
  'contrib':'What is raising your risk?',
  'rec_title':'What should you do now?',
  'est_lbl':'Estimated probability',
  'layer_lbl':'Estimate',
  'disc':'This tool predicts hypertension risk and generates recommendations and does not replace a clinical diagnosis. Blood pressure measurement by a qualified health professional is required.',
},
'rw':{
  'title':'Igikoresho cyo Gusuzuma Ibyago bya Hypertension',
  'tab1':'AMAKURU YAWE','tab2':'ONGERAHO IBIPIMO',
  'l1':"Subiza ibibazo bikurikira kugira ngo ubone igipimo cy'ibyago byawe bya hypertension.",
  'l2':"Ongeraho ibipimo by'umubiri kugira ngo ibisubizo bibe nziza.",
  'add_l1':"Soza ubutaka bwa mbere bw'Amakuru Yawe.",
  'about':'Amakuru Yawe','life':"Imyitwarire Yawe",
  'body':"Ibipimo by'Umubiri",
  'age':'Ufite imyaka ingahe?','sex':'Igitsina','prov':'Intara',
  'urb':'Utuye he?','educ':'Amashuri warangije',
  'pi':'Mukora imyitozo ngorora mubiri?',
  'rur':'Icyaro','urbn':'Umujyi',
  'e1':'Ntayo cyangwa Abanza','e2':'Ayisumbuye','e3':'Kaminuza cyangwa hejuru',
  'alc':'Wanywa inzoga?',
  'd5':'Kangahe wongeraho umunyu mu biryo?',
  'd5_help':"Umunyu wongeragaho ku meza cyangwa mu guteka",
  'd7':'Nikangahe urya ibiryo byo mu nganda bibamo umunyu mwinshi?',
  'd7_help':"Urugero: chips, ibiryo bifungirwa, soseji, noodles",
  'diab':'Muganga yakubwiye ko ufite Diabete?',
  'diab_yes_badge':"Sukari yongera ibyago by'umuvuduko",
  'wt':'Uburemere (kg)','ht':'Uburebure (cm)',
  'wst':"Ingano y'ikibuno (cm)",
  'wst_check':"Nasuye ingano y'ikibuno cyanjye",
  'wst_help':"Sura ku rwego rw'inda yawe ukoresheje metero",
  'wst_skip':"Ingano y'ikibuno ntisuzumwe — izabarwa bikurikije ibindi bipimo",
  'recalc_note':"Niba wahinduje igisubizo icyo aricyo cyose haruguru, kanda 'Bara Ibyago Byange' nanone kugira ngo ubone ibisubizo bishya.",
  'btn1':'Bara Ibyago Byange',
  'btn2':'Hindura Ibisubizo Byange',
  'reset':'Tangira Bushya',
  'sel_model':'Hitamo Indangagaciro',
  'bmi_normal':'Uburemere busanzwe','bmi_over':'Uburemere burenga','bmi_obese':'Ubushyohe',
  'wst_normal':'Bisanzwe','wst_high':'Irenga umurego',
  'pa':'Yego',
  'pa_badge':"Nibura iminota 30 munsi 5 y'icyumweru",
  'pi_':'Oya',
  'pi_badge':"Ntabwo ngeza iminota 30 munsi 5 y'icyumweru",
  'a0':'Ntabwo nywa inzoga',
  'a1':'Nayinyweye umwaka ushize',
  'a2':'Nayinyweye mu kwezi gushize',
  's1':'Ntanarimwe',
  's2':'Gake  (Iminsi 1–3 mu kwezi)',
  's3':'Rimwe na rimwe  (Iminsi 2–3 mu cyumweru)',
  's4':'Akenshi  (Iminsi 4–5 mu cyumweru)',
  's5':'Buri munsi',
  'pe':'Intara y\'Uburasirazuba','pk':'Umujyi wa Kigali',
  'pn':'Intara y\'Amajyaruguru','ps':'Intara y\'Amajyepfo','pw':'Intara y\'Uburengerazuba',
  'fem':'Gore','mal':'Gabo',
  'no':'Oya','yes':'Yego',
  'low':'Ibyago Bike','mod':'Ibyago Biri Hagati',
  'high':'Ibyago Bikabije','vhi':'Ibyago Bikabije Cyane',
  'contrib':"Ni iki gituma hari ibyago nk'aya?",
  'rec_title':'Ugomba gukora iki ubu?',
  'est_lbl':'Ibyago bisuzumwe',
  'layer_lbl':'Igipimo',
  'disc':"Iki gikoresho gipima ibyago bya hypertension kandi gitanga inama. Ntigisimbuza isuzuma ry'umuganga. Hypertension igomba kwemezwa n'inzobere z'ubuzima.",
}}

# ── Session state ─────────────────────────────────────────────────────────────
DEFS = {
    'lang':'en','layers':0,'assessed':False,
    'prob':0.0,'pd_data':{},'show_met':False,
}
for k,v in DEFS.items():
    if k not in st.session_state: st.session_state[k]=v

params = st.query_params
if 'lang' in params:
    nl = params['lang']
    if nl in ('en','rw') and nl != st.session_state['lang']:
        st.session_state['lang'] = nl
        st.rerun()

L = st.session_state['lang']
T = TX[L]

# ── Load artifacts ────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_all():
    mice      = joblib.load(os.path.join(BASE,'mice_imputer_v2.pkl'))
    pt        = joblib.load(os.path.join(BASE,'power_transformer_v2.pkl'))
    ohe_prov  = joblib.load(os.path.join(BASE,'encoder_province_v2.pkl'))
    ohe_c7    = joblib.load(os.path.join(BASE,'encoder_c7_v2.pkl'))
    xgb_model = joblib.load(os.path.join(BASE,'xgb_deploy_p3.pkl'))
    cat_model = joblib.load(os.path.join(BASE,'cat_deploy_v2.pkl'))
    mice_feats= joblib.load(os.path.join(BASE,'mice_features_v2.pkl'))
    pt_feats  = joblib.load(os.path.join(BASE,'pt_features_v2.pkl'))
    return mice,pt,ohe_prov,ohe_c7,xgb_model,cat_model,mice_feats,pt_feats

try:
    mice,pt,ohe_prov,ohe_c7,xgb_model,cat_model,mice_feats,pt_feats = load_all()
    ok = True
except Exception as e:
    ok = False; err_msg = str(e)

FEATS_XGB = [
    'age','urbanrural_new','educ','total_MET','alcohol_status',
    'd7','d5','bmi','m14','hipwaistrate','average_heartrate',
    'b8','b5','b17','sodium','creatinine','h17a','h1','h6',
    'age_x_female','bmi_x_alcohol','age_x_bmi','sodium_x_bmi',
    'province_North','province_South','province_West',
]

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(v, use_xgb=False):
    bmi  = v.get('bmi',  np.nan); m14  = v.get('m14',  np.nan)
    hr   = v.get('hr',   np.nan); b8   = v.get('b8',   np.nan)
    b5   = v.get('b5',   np.nan); b17  = v.get('b17',  np.nan)
    sod  = v.get('sodium', np.nan); creat= v.get('creatinine', np.nan)
    alc  = v.get('alc', 0)
    is_f = 1.0 if v.get('sex',2)==2 else 0.0

    row = {
        'age':v['age'],'c1':v.get('sex',2),
        'province_new':v['prov'],'urbanrural_new':v.get('urb',1),
        'educ':v.get('educ',1),'c7':v.get('mar',1),
        'pi':v.get('pi',0),'total_MET':v.get('met',5000),
        'alcohol_status':float(alc),'d7':v.get('d7',3),'d5':v.get('d5',3),
        'bmi':bmi,'m14':m14,'hipwaistrate':np.nan,'average_heartrate':hr,
        'b8':b8,'b5':b5,'b17':b17,'sodium':sod,'creatinine':creat,
        'h17a':v.get('h17a',0),'h1':v.get('h1',0),
        'h6':v.get('h6',0),'h18':v.get('h18',0),
        'age_x_female':v['age']*is_f,
        'bmi_x_alcohol':np.nan,'age_x_bmi':np.nan,
        'sodium_x_bmi':np.nan,'age_x_alcohol':np.nan,
        'age_x_heartrate':np.nan,'m14_x_alcohol':np.nan,'bmi_x_sodium':np.nan,
    }
    df_in  = pd.DataFrame([row])[mice_feats]
    df_imp = pd.DataFrame(mice.transform(df_in), columns=mice_feats)

    # DEFINITIVE FIX: After MICE, unconditionally restore
    # EVERY variable the user actually measured.
    # MICE must only fill truly unknown variables.
    # For any variable where user provided a value:
    #   → ignore MICE output, use actual value
    # This prevents any cross-variable MICE distortion.
    measured = {
        'bmi':              bmi,
        'm14':              m14,
        'average_heartrate':hr,
        'sodium':           sod,
        'b8':               b8,
        'b5':               b5,
        'b17':              b17,
        'creatinine':       creat,
    }
    for col, val in measured.items():
        if not np.isnan(val):
            df_imp[col] = val   # overwrite MICE with actual

    # Use actual values for interactions (not MICE estimates)
    bmi_r = bmi   if not np.isnan(bmi)  else df_imp['bmi'].values[0]
    m14_r = m14   if not np.isnan(m14)  else df_imp['m14'].values[0]
    hr_r  = hr    if not np.isnan(hr)   else df_imp['average_heartrate'].values[0]
    sod_r = sod   if not np.isnan(sod)  else df_imp['sodium'].values[0]
    age_r = float(v['age'])

    df_imp['alcohol_status']  = float(alc)
    df_imp['bmi_x_alcohol']   = bmi_r*alc
    df_imp['age_x_bmi']       = age_r*bmi_r
    df_imp['sodium_x_bmi']    = sod_r*bmi_r
    df_imp['age_x_alcohol']   = age_r*alc
    df_imp['age_x_heartrate'] = age_r*hr_r
    df_imp['m14_x_alcohol']   = m14_r*alc
    df_imp['bmi_x_sodium']    = bmi_r*sod_r

    df_imp['province_new'] = df_imp['province_new'].round().clip(1,5).astype(int)
    df_imp['c7']           = df_imp['c7'].round().clip(1,3).astype(int)
    prov_lbl={1:'East',2:'Kigali',3:'North',4:'South',5:'West'}
    c7_lbl  ={1:'Single',2:'Married',3:'Widowed'}
    pa=ohe_prov.transform(df_imp[['province_new']])
    ca=ohe_c7.transform(df_imp[['c7']])
    for i,n in enumerate([f'province_{prov_lbl[int(c)]}' for c in ohe_prov.categories_[0].tolist()[1:]]):
        df_imp[n]=pa[:,i]
    for i,n in enumerate([f'c7_{c7_lbl[int(c)]}' for c in ohe_c7.categories_[0].tolist()[1:]]):
        df_imp[n]=ca[:,i]
    df_imp.drop(columns=['province_new','c7','is_female'],errors='ignore',inplace=True)
    pt_p=[c for c in pt_feats if c in df_imp.columns]
    df_imp[pt_p]=pt.transform(df_imp[pt_p])
    X=df_imp.reindex(columns=FEATS_XGB,fill_value=0).values
    m=xgb_model if use_xgb else cat_model
    return float(np.clip(m.predict_proba(X)[0,1],0.01,0.99))

# ── Clinical alerts ───────────────────────────────────────────────────────────
def get_alerts(v, L):
    alerts = []
    bmi_v = v.get('bmi', None)
    wst_v = v.get('m14', None)
    hr_v  = v.get('hr',  None)
    b8_v  = v.get('b8',  None)
    b5_v  = v.get('b5',  None)
    alc_v = v.get('alc', 0)
    pi_v  = v.get('pi',  0)
    d5_v  = v.get('d5',  3)
    d7_v  = v.get('d7',  3)
    sex_v = v.get('sex', 2)
    age_v = v.get('age', 40)
    prov_v= v.get('prov', 1)
    h17a_v= v.get('h17a', 0)

    if L=='en':
        if bmi_v:
            if bmi_v >= 30:
                alerts.append(('Your BMI is high', f'BMI {bmi_v:.1f} kg/m² (Obese)', 'Obesity is a major risk factor for hypertension and heart disease.', 'alert'))
            elif bmi_v >= 25:
                alerts.append(('Your BMI is above normal', f'BMI {bmi_v:.1f} kg/m² (Overweight)', 'Weight reduction can help lower your blood pressure.', 'warn'))
        if wst_v:
            cutoff = 102 if sex_v==1 else 88
            label  = '102 cm for men' if sex_v==1 else '88 cm for women'
            if wst_v > cutoff:
                alerts.append(('Your waist is above the safe limit', f'{wst_v:.0f} cm (limit is {label})', 'Excess abdominal fat is strongly linked to high blood pressure.', 'alert'))
        if b5_v and b5_v >= 126:
            alerts.append(('You have diabetes', 'Confirmed', 'Diabetes and hypertension frequently occur together. Controlling blood sugar also protects blood pressure.', 'alert'))
        if alc_v == 2:
            alerts.append(('You are currently drinking alcohol', 'Past 30 days', 'Regular alcohol consumption raises blood pressure over time.', 'alert'))
        elif alc_v == 1:
            alerts.append(('You have recently consumed alcohol', 'Past 12 months', 'Alcohol use is associated with increased blood pressure risk.', 'warn'))
        if pi_v == 1:
            alerts.append(('Your physical activity level is low', 'Below WHO recommendation', 'Regular physical activity is one of the most effective ways to reduce blood pressure.', 'warn'))
        if d5_v >= 4:
            alerts.append(('Your salt intake is high', 'Often or always adding salt', 'Reducing dietary salt is one of the most effective ways to lower blood pressure.', 'alert'))
        if d7_v >= 4:
            alerts.append(('You frequently eat processed food', 'Often or always', 'Processed foods contain high levels of hidden salt which raises blood pressure.', 'alert'))
        if prov_v == 4:
            alerts.append(('You are in Southern Province', 'Highest hypertension rate nationally', 'Southern Province has the highest hypertension prevalence in Rwanda.', 'warn'))
        if prov_v == 5:
            alerts.append(('You are in Western Province', 'Above average hypertension rate', 'Western Province has above-average hypertension prevalence in Rwanda.', 'warn'))
        if age_v >= 55:
            alerts.append(('Your age is a significant risk factor', f'{age_v} years old', 'Age is the strongest single risk factor for hypertension.', 'alert'))
        elif age_v >= 45:
            alerts.append(('Your age increases your risk', f'{age_v} years old', 'Hypertension risk rises significantly after the age of 45.', 'warn'))
    else:
        if bmi_v:
            if bmi_v >= 30:
                alerts.append(('Uburemere bwawe burenga cyane', f'BMI {bmi_v:.1f} kg/m² (Ubushyohe)', "Ubushyohe ni impamvu nkuru y'umuvuduko w'amaraso.", 'alert'))
            elif bmi_v >= 25:
                alerts.append(('Uburemere bwawe burenga gato', f'{bmi_v:.1f} kg/m²', 'Kugabanya ibiro bizagabanya umuvuduko.', 'warn'))
        if wst_v:
            cutoff = 102 if sex_v==1 else 88
            if wst_v > cutoff:
                alerts.append(('Ikibuno cyawe kirenga umurego', f'{wst_v:.0f} cm', "Uburemere bw'ikibuno buhuriye n'umuvuduko w'amaraso.", 'alert'))
        if b5_v and b5_v >= 126:
            alerts.append(('Ufite Diabete', 'Yemejwe', "Diabete n'umuvuduko bikunze kujyana. Kugenzura Sukari birarinda umuvuduko.", 'alert'))
        if alc_v == 2:
            alerts.append(('Unywa inzoga', 'Mu kwezi gushize', "Inzoga zongera umuvuduko w'amaraso.", 'alert'))
        elif alc_v == 1:
            alerts.append(('Wanyweye inzoga vuba', 'Mu mwaka ushize', "Inzoga zihuriye n'ibyago by'umuvuduko.", 'warn'))
        if pi_v == 1:
            alerts.append(('Imyitozo yawe ntihagije', 'Munsi y\'ibisabwa na WHO', "Imyitozo igabanya umuvuduko w'amaraso.", 'warn'))
        if d5_v >= 4:
            alerts.append(('Urya umunyu mwinshi', 'Akenshi cyangwa buri munsi', 'Kugabanya umunyu ni umwe mu buryo bwiza bwo kugabanya umuvuduko.', 'alert'))
        if d7_v >= 4:
            alerts.append(('Urya ibiryo byungutse kenshi', 'Akenshi cyangwa buri munsi', 'Ibiryo byungutse birimo umunyu mwinshi uziguye ungana n\'umuvuduko.', 'alert'))
        if age_v >= 55:
            alerts.append(('Imyaka yawe ni impamvu ikomeye', f'Imyaka {age_v}', 'Imyaka ni impamvu nkuru ya hypertension.', 'alert'))
        elif age_v >= 45:
            alerts.append(('Imyaka yawe yongera ibyago', f'Imyaka {age_v}', 'Ibyago bya hypertension biyongera nyuma y\'imyaka 45.', 'warn'))
    return alerts

# ── Gauge ─────────────────────────────────────────────────────────────────────
def gauge(prob, threshold):
    pct = round(prob*100,1)
    if prob < 0.25:     color='#1A7A3C'; zone='LOW'
    elif prob < threshold: color='#B7770D'; zone='MOD'
    elif prob < threshold+0.20: color='#B94A00'; zone='HIGH'
    else:               color='#922B21'; zone='VHIGH'
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix':'%','font':{'size':46,'family':'DM Serif Display','color':color}},
        gauge={
            'axis':{'range':[0,100],'tickwidth':1,'tickcolor':'#8A9BAB',
                    'tickvals':[0,25,int(threshold*100),int((threshold+0.20)*100),100],
                    'tickfont':{'size':9,'family':'DM Sans'}},
            'bar':{'color':color,'thickness':0.28},
            'bgcolor':'white','borderwidth':0,
            'steps':[
                {'range':[0,25],                        'color':'#C8EDD4'},
                {'range':[25,threshold*100],            'color':'#FAF0C0'},
                {'range':[threshold*100,(threshold+0.20)*100],'color':'#F5CBA7'},
                {'range':[(threshold+0.20)*100,100],    'color':'#F1948A'},
            ],
            'threshold':{'line':{'color':color,'width':3},'thickness':0.82,'value':pct}
        }
    ))
    fig.update_layout(height=200,margin=dict(l=30,r=30,t=10,b=5),
                      paper_bgcolor='white',font_family='DM Sans')
    return fig, color, zone

# ── Show risk result ──────────────────────────────────────────────────────────
def show_result(prob, threshold, layer_num, use_xgb, v, L, T):
    fig, color, zone = gauge(prob, threshold)
    pct   = round(prob*100,1)
    mname = "CatBoost" if not use_xgb else "XGBoost"
    labels= {'LOW':T['low'],'MOD':T['mod'],'HIGH':T['high'],'VHIGH':T['vhi']}
    cls   = {'LOW':'risk-low','MOD':'risk-mod','HIGH':'risk-high','VHIGH':'risk-vhi'}
    actions_en = {
        'LOW': [
            ('Annual blood pressure check', 'Visit your nearest health facility once a year to have your blood pressure measured.', '#1A7A3C'),
            ('Maintain a healthy lifestyle', 'Continue regular physical activity and a balanced low-salt diet.', '#1A7A3C'),
            ('Reduce dietary salt', 'WHO recommends less than 5g of sodium per day. Avoid adding salt at the table or during cooking.', '#1A7A3C'),
            ('Limit processed and packaged food', 'Processed foods contain hidden salt. Limit chips, tinned food, sausages and instant noodles.', '#1A7A3C'),
            ('Increase physical activity', 'Aim for at least 30 minutes of moderate exercise on 5 or more days per week.', '#1A7A3C'),
            ('Reduce or stop alcohol consumption', 'Alcohol raises blood pressure over time. Reducing consumption reduces your risk.', '#1A7A3C'),
            ('Achieve and maintain healthy weight', 'Losing 5 to 10 percent of body weight can reduce blood pressure by up to 5 mmHg.', '#1A7A3C'),
            ('Control your blood sugar', 'If you have diabetes, careful blood sugar control also protects blood pressure.', '#1A7A3C'),
        ],
        'MOD': [
            ('Blood pressure check within 6 months', 'Visit a health centre for blood pressure measurement within the next 6 months.', '#B7770D'),
            ('Reduce dietary salt', 'WHO recommends less than 5g of sodium per day. Avoid adding salt at the table or during cooking.', '#B7770D'),
            ('Limit processed and packaged food', 'Processed foods contain hidden salt. Limit chips, tinned food, sausages and instant noodles.', '#B7770D'),
            ('Increase physical activity', 'Aim for at least 30 minutes of moderate exercise on 5 or more days per week.', '#B7770D'),
        ],
        'HIGH': [
            ('Blood pressure check within 2 weeks', 'Visit a health centre promptly. If blood pressure is 140/90 mmHg or above, treatment should be initiated.', '#B94A00'),
            ('Limit salt and processed food', 'Restrict dietary sodium to less than 5g per day. Avoid chips, tinned food, sausages and instant noodles.', '#B94A00'),
            ('Reduce or stop alcohol consumption', 'Alcohol directly raises blood pressure. Reducing consumption has measurable benefit within weeks.', '#B94A00'),
            ('Increase physical activity', '30 minutes of brisk walking or equivalent activity, 5 days per week.', '#B94A00'),
            ('Achieve and maintain healthy weight', 'Losing 5 to 10 percent of body weight can reduce blood pressure by up to 5 mmHg.', '#B94A00'),
            ('Control your blood sugar', 'If you have diabetes, careful blood sugar control is essential — diabetes and hypertension together significantly increase cardiovascular risk.', '#B94A00'),
        ],
        'VHIGH': [
            ('Go to a health facility today', 'Visit the nearest health centre immediately for blood pressure measurement. Do not delay.', '#922B21'),
            ('Seek medical treatment today', 'If blood pressure is 160/100 mmHg or above, antihypertensive treatment must be started today.', '#922B21'),
            ('Stop or reduce alcohol immediately', 'Alcohol directly raises blood pressure. Stopping or reducing consumption will help immediately.', '#922B21'),
            ('Strict dietary sodium restriction', 'No added salt. Avoid all processed food. Restrict sodium to less than 5g per day starting today.', '#922B21'),
            ('Urgent blood sugar control', 'If you have diabetes, urgent management is required — hypertension and diabetes together are life-threatening.', '#922B21'),
        ],
    }
    actions_rw = {
        'LOW': [
            ("Gupima umuvuduko buri mwaka", "Genda ku ivuriro ryegereye buri mwaka gupimuza umuvuduko w'amaraso.", '#1A7A3C'),
            ("Komeza imyitwarire myiza", "Komeza imyitozo n'indyo iringaniye irimo umunyu muke.", '#1A7A3C'),
            ("Gabanya umunyu mu biryo", "WHO isaba munsi ya 5g buri munsi. Ntiwogeraho umunyu ku meza cyangwa mu guteka.", '#1A7A3C'),
            ("Irinde ibiryo byo mu nganda", "Ibiryo byo mu nganda birimo umunyu mwinshi. Irinde chips, ibiryo bifungirwa, soseji na noodles.", '#1A7A3C'),
            ("Ongera imyitozo", "Kora imyitozo nibura iminota 30 ku minsi 5 cyangwa irenga mu cyumweru.", '#1A7A3C'),
            ("Gabanya cyangwa hagarika inzoga", "Inzoga zongera umuvuduko. Kuzigabanya bigabanya ibyago.", '#1A7A3C'),
            ("Genzura ibiro byawe", "Kugabanya ibiro bikabije bigabanya umuvuduko w'amaraso.", '#1A7A3C'),
            ("Genzura sukari mu maraso", "Niba ufite Diabete, genzura neza sukari kuko birarinda umuvuduko.", '#1A7A3C'),
        ],
        'MOD': [
            ("Gupima umuvuduko mu mezi 6", "Genda ku ivuriro gupimuza umuvuduko w'amaraso mu mezi 6 iri imbere.", '#B7770D'),
            ("Gabanya umunyu mu biryo", "WHO isaba munsi ya 5g buri munsi. Ntiwogeraho umunyu ku meza cyangwa mu guteka.", '#B7770D'),
            ("Irinde ibiryo byo mu nganda", "Ibiryo byo mu nganda birimo umunyu mwinshi. Irinde chips, ibiryo bifungirwa, soseji na noodles.", '#B7770D'),
            ("Ongera imyitozo", "Kora imyitozo nibura iminota 30 ku minsi 5 cyangwa irenga mu cyumweru.", '#B7770D'),
        ],
        'HIGH': [
            ("Gupima umuvuduko mu cyumweru 2", "Genda ku ivuriro vuba. Niba umuvuduko uri kuri 140/90 mmHg cyangwa hejuru, ugomba gutangira kuvurwa.", '#B94A00'),
            ("Gabanya umunyu n'ibiryo byungutse", "Gabanya umunyu munsi ya 5g buri munsi. Irinde chips, ibiryo bifungirwa, soseji na noodles.", '#B94A00'),
            ("Gabanya cyangwa hagarika inzoga", "Inzoga zongera umuvuduko w'amaraso. Kuzigabanya bigira akamaro mu byumweru bike.", '#B94A00'),
            ("Ongera imyitozo", "Iminota 30 yo kugenda vuba cyangwa imyitozo indi, inshuro 5 mu cyumweru.", '#B94A00'),
            ("Genzura ibiro byawe", "Kugabanya ibiro bikabije bigabanya umuvuduko w'amaraso.", '#B94A00'),
            ("Genzura sukari mu maraso", "Niba ufite Diabete, genzura neza sukari — Diabete n'umuvuduko bitera ingaruka zikomeye ku mutima.", '#B94A00'),
        ],
        'VHIGH': [
            ("Genda ku ivuriro uyu munsi", "Genda ku ivuriro vuba gupimuza umuvuduko w'amaraso. Ntugomba gutinda.", '#922B21'),
            ("Shakisha imiti uyu munsi", "Niba umuvuduko uri kuri 160/100 mmHg cyangwa hejuru, ugomba gutangira imiti uyu munsi.", '#922B21'),
            ("Hagarika inzoga none", "Inzoga zongera umuvuduko w'amaraso. Kuzizigama bigira akamaro vuba.", '#922B21'),
            ("Gabanya umunyu bikabije", "Ntiwogeraho umunyu. Irinde ibiryo byungutse byose. Munsi ya 5g buri munsi uhereye uyu munsi.", '#922B21'),
            ("Genzura sukari vuba", "Niba ufite Diabete, genzura sukari vuba — Diabete n'umuvuduko bikabije birateza akaga ku buzima.", '#922B21'),
        ],
    }
    actions = actions_en if L=='en' else actions_rw
    alerts  = get_alerts(v, L)

    # Build dynamic recommendation list based on alerts triggered
    act_list = []
    alert_names = [a[0].lower() for a in alerts]
    base = actions.get(zone, actions['LOW'])

    for item in base:
        title_l = item[0].lower()
        # Always include BP check
        if 'bp' in title_l or 'gupima' in title_l or 'go to' in title_l or 'ivuriro' in title_l:
            act_list.append(item)
        # Salt — only if salt or processed food alert triggered
        elif 'salt' in title_l or 'umunyu' in title_l:
            if any('salt' in a or 'processed' in a or 'umunyu' in a or 'byungutse' in a
                   for a in alert_names):
                act_list.append(item)
        # Alcohol — only if alcohol alert triggered
        elif 'alcohol' in title_l or 'inzoga' in title_l:
            if any('alcohol' in a or 'inzoga' in a for a in alert_names):
                act_list.append(item)
        # Activity — only if inactivity alert triggered
        elif 'active' in title_l or 'imyitozo' in title_l:
            if any('exercise' in a or 'imyitozo' in a for a in alert_names):
                act_list.append(item)
        # Weight — only if BMI alert triggered
        elif 'weight' in title_l or 'ibiro' in title_l:
            if any('bmi' in a or 'uburemere' in a for a in alert_names):
                act_list.append(item)
        # Diabetes — only if diabetes alert triggered
        elif 'diabetes' in title_l or 'diabete' in title_l or 'sukari' in title_l:
            if any('diabetes' in a or 'sukari' in a or 'diabete' in a
                   for a in alert_names):
                act_list.append(item)
        # Province — always include awareness rec
        elif 'province' in title_l or 'intara' in title_l:
            if any('province' in a or 'intara' in a for a in alert_names):
                act_list.append(item)
        # Age — always include awareness rec
        elif 'age' in title_l or 'imyaka' in title_l:
            if any('age' in a or 'imyaka' in a for a in alert_names):
                act_list.append(item)
        # Everything else include
        else:
            act_list.append(item)

    # Fallback — always show at least 2 items
    if len(act_list) < 2:
        act_list = base[:2]

    # Quality dots
    lyr_names_en = ['','Lifestyle estimate','Refined with measurements']
    lyr_names_rw = ['','Igipimo cy\'imyitwarire','Bibonetse n\'ibipimo']
    lyr_names = lyr_names_en if L=='en' else lyr_names_rw
    dots = ''.join([f'<div class="qdot {"on" if i<layer_num else ""}"></div>' for i in range(2)])
    st.markdown(
        f'<div class="qlbl">{T["layer_lbl"]} {layer_num} — {lyr_names[layer_num]}</div>'
        f'<div class="qdots">{dots}</div>',
        unsafe_allow_html=True)

    # Gauge
    st.markdown('<div class="card" style="padding:10px 16px">', unsafe_allow_html=True)
    st.plotly_chart(fig, width='stretch',
                    key=f'gauge_l{layer_num}', config={'displayModeBar':False})

    # Risk banner
    msg_en = {'LOW':'Annual blood pressure check recommended. Maintain a healthy lifestyle.',
              'MOD':'Schedule blood pressure measurement within the next 6 months.',
              'HIGH':'Visit a health centre within 2 weeks for blood pressure measurement.',
              'VHIGH':'Urgent blood pressure measurement required today. Do not delay.'}
    msg_rw = {'LOW':"Pima umuvuduko buri mwaka. Komeza imyitwarire myiza.",
              'MOD':"Teganya gupima umuvuduko mu mezi 6 iri imbere.",
              'HIGH':"Genda ku ivuriro mu cyumweru 2 gupimuza umuvuduko.",
              'VHIGH':"Genda gupimuza umuvuduko w'amaraso uyu munsi. Ntugomba gutinda."}
    msg    = msg_en[zone] if L=='en' else msg_rw[zone]
    st.markdown(f"""
    <div class="risk-banner {cls[zone]}">
      <div class="risk-label" style="color:{color}">{labels[zone]}</div>
      <div class="risk-pct"   style="color:{color}">{pct}%</div>
      <div class="rbar-bg"><div class="rbar" style="width:{pct}%;background:{color}"></div></div>
      <div class="risk-msg">{msg}</div>
      <div class="risk-meta">{mname} · Threshold {threshold} · {T['est_lbl']}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Clinical alerts
    if alerts:
        st.markdown(f'<div class="slbl">{T["contrib"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        colors_map = {'alert':'#922B21','warn':'#B7770D','ok':'#1A7A3C'}
        badge_cls  = {'alert':'abadge-alert','warn':'abadge-warn','ok':'abadge-ok'}
        badge_txt  = {'alert':'HIGH' if L=='en' else 'HEJURU',
                      'warn':'MODERATE' if L=='en' else 'HAGATI',
                      'ok':'NORMAL'}
        rows = ''
        for name, val, desc, lvl in alerts:
            c = colors_map[lvl]
            b = f'<span class="abadge {badge_cls[lvl]}">{badge_txt[lvl]}</span>'
            rows += f"""
            <div class="alert-item">
              <div class="alert-dot" style="background:{c}"></div>
              <div>
                <div class="alert-name">{name} <span style="font-weight:400;color:var(--text2)">({val})</span>{b}</div>
                <div class="alert-val">{desc}</div>
              </div>
            </div>"""
        st.markdown(rows, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Recommended actions — dynamic based on alerts
    st.markdown(f'<div class="slbl">{T["rec_title"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    rows = ''
    for i,(title,desc,c) in enumerate(act_list):
        rows += f"""
        <div class="rec-item">
          <div style="display:flex;align-items:flex-start;gap:10px">
            <div class="rec-num" style="background:{c}">{i+1}</div>
            <div>
              <div class="rec-title">{title}</div>
              <div class="rec-desc">{desc}</div>
            </div>
          </div>
        </div>"""
    st.markdown(rows, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hdr-blue">
  <div class="hdr-title">{T['title']}</div>
</div>
<div class="hdr-yellow"></div>
<div class="hdr-green">
  <div class="lang-row">
    <a class="lbtn {'on' if L=='en' else ''}" href="?lang=en">English</a>
    <a class="lbtn {'on' if L=='rw' else ''}" href="?lang=rw">Kinyarwanda</a>
  </div>
</div>
""", unsafe_allow_html=True)

if not ok:
    st.error(f"Model files not found in: {BASE}")
    st.code(err_msg); st.stop()

# ── Model — CatBoost v2 fixed for public deployment ──────────────────────────
USE_XGB   = False
THRESHOLD = 0.39

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([T['tab1'], T['tab2']])

# ════ LAYER 1 ════
with tab1:
    st.caption(T['l1'])

    # ── ABOUT YOU — 2x2 grid (4 vars) ────────────────────────────────────────
    st.markdown(f'<div class="slbl">{T["about"]}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age  = st.number_input(T['age'], 18, 69, 40, key='k_age')
        sex  = st.radio(T['sex'], [T['fem'],T['mal']], horizontal=True, key='k_sex')
    with c2:
        prov = st.selectbox(T['prov'], [T['pe'],T['pk'],T['pn'],T['ps'],T['pw']], key='k_prov')
        urb  = st.radio(T['urb'], [T['rur'],T['urbn']], horizontal=True, key='k_urb')

    # Education — full width (odd var out)
    educ = st.selectbox(T['educ'], [T['e1'],T['e2'],T['e3']], key='k_educ')

    # ── YOUR LIFESTYLE ────────────────────────────────────────────────────────
    st.markdown(f'<div class="slbl">{T["life"]}</div>', unsafe_allow_html=True)

    # 2x2 grid — PA+Alcohol+Diabetes left | Salt+Processed right
    c3, c4 = st.columns(2)
    with c3:
        # Physical Activity
        st.markdown(f'<p style="font-size:11px;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;color:var(--text2);margin-bottom:6px">{T["pi"]}</p>', unsafe_allow_html=True)
        pa_btn = st.radio(T['pi'], [T['pa'], T['pi_']], key='k_pi',
                          label_visibility='collapsed', horizontal=True)
        if pa_btn == T['pa']:
            st.markdown(f'''
            <div style="display:inline-flex;align-items:center;gap:6px;
                        background:#EAF7EF;border:1.5px solid #1A7A3C;border-radius:6px;
                        padding:5px 12px;margin-bottom:10px">
              <span style="color:#1A7A3C;font-size:12px;font-weight:700">YES</span>
              <span style="font-size:11px;font-weight:600;color:#145A2E">{T["pa_badge"]}</span>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div style="display:inline-flex;align-items:center;gap:6px;
                        background:#FDEDEC;border:1.5px solid #922B21;border-radius:6px;
                        padding:5px 12px;margin-bottom:10px">
              <span style="color:#922B21;font-size:12px;font-weight:700">NO</span>
              <span style="font-size:11px;font-weight:600;color:#922B21">{T["pi_badge"]}</span>
            </div>''', unsafe_allow_html=True)
        # Alcohol
        alc = st.selectbox(T['alc'], [T['a0'],T['a1'],T['a2']], key='k_alc')
        # Diabetes — under alcohol in left column
        diab = st.radio(T['diab'], [T['no'],T['yes']], horizontal=True, key='k_diab')
        if diab == T['yes']:
            st.markdown(f'''
            <div style="display:inline-flex;align-items:center;gap:6px;
                        background:#FDEDEC;border:1.5px solid #922B21;border-radius:6px;
                        padding:5px 12px;margin-top:2px;margin-bottom:6px">
              <span style="color:#922B21;font-size:12px;font-weight:700">NOTE</span>
              <span style="font-size:11px;font-weight:600;color:#922B21">{T["diab_yes_badge"]}</span>
            </div>''', unsafe_allow_html=True)

    with c4:
        # Salt — hint above
        st.caption(T['d5_help'])
        d5 = st.selectbox(T['d5'], [T['s1'],T['s2'],T['s3'],T['s4'],T['s5']], index=2, key='k_d5')
        # Processed — hint above
        st.caption(T['d7_help'])
        d7 = st.selectbox(T['d7'], [T['s1'],T['s2'],T['s3'],T['s4'],T['s5']], index=2, key='k_d7')

    # ── Buttons — CALCULATE + RESET same row ─────────────────────────────────
    st.markdown("""<style>
    div[data-testid="column"]:last-child .stButton>button{
      margin-top:0!important;height:100%;width:100%}
    div[data-testid="column"]:first-child .stButton>button{
      width:100%}
    </style>""", unsafe_allow_html=True)
    cb1, cb2 = st.columns(2)
    with cb1:
        btn1_clicked = st.button(T['btn1'], key='btn1')
    with cb2:
        if st.button(T['reset'], key='btn_reset'):
            for k in ['layers','assessed','prob','pd_data']:
                st.session_state[k] = DEFS[k]
            st.rerun()

    if btn1_clicked:
        pm  = {T['pe']:1,T['pk']:2,T['pn']:3,T['ps']:4,T['pw']:5}
        am  = {T['a0']:0,T['a1']:1,T['a2']:2}
        sm  = {T['s1']:1,T['s2']:2,T['s3']:3,T['s4']:4,T['s5']:5}
        em  = {T['e1']:1,T['e2']:2,T['e3']:3}
        urb_map = {T['rur']:1,T['urbn']:2}
        b5_val  = 130.0 if diab==T['yes'] else np.nan
        vals = dict(age=int(age), sex=2 if sex==T['fem'] else 1,
                    prov=pm[prov], urb=urb_map.get(urb,1),
                    educ=em.get(educ,1), mar=1,
                    pi=1 if pa_btn==T['pi_'] else 0,
                    met=5000, alc=am[alc],
                    d5=sm[d5], d7=sm[d7],
                    b5=b5_val,
                    h17a=0, h1=0, h6=0, h18=0)
        prob = predict(vals, USE_XGB)
        st.session_state.update({'prob':prob,'layers':1,'assessed':True,'pd_data':vals})

    if st.session_state['assessed'] and st.session_state['layers']>=1:
        pm  = {T['pe']:1,T['pk']:2,T['pn']:3,T['ps']:4,T['pw']:5}
        am  = {T['a0']:0,T['a1']:1,T['a2']:2}
        sm  = {T['s1']:1,T['s2']:2,T['s3']:3,T['s4']:4,T['s5']:5}
        em  = {T['e1']:1,T['e2']:2,T['e3']:3}
        urb_map = {T['rur']:1,T['urbn']:2}
        b5_val  = 130.0 if diab==T['yes'] else np.nan
        live_vals = dict(age=int(age), sex=2 if sex==T['fem'] else 1,
                         prov=pm[prov], urb=urb_map.get(urb,1),
                         educ=em.get(educ,1), mar=1,
                         pi=1 if pa_btn==T['pi_'] else 0,
                         met=5000, alc=am[alc],
                         d5=sm[d5], d7=sm[d7],
                         b5=b5_val,
                         h17a=0, h1=0, h6=0, h18=0)
        merged = dict(st.session_state['pd_data'])
        merged.update(live_vals)
        show_result(st.session_state['prob'],THRESHOLD,1,USE_XGB,merged,L,T)
        st.markdown(f"""
        <div style="background:#EBF5FB;border-left:3px solid #2980B9;
                    border-radius:4px;padding:10px 14px;margin:8px 0">
          <div style="font-size:12px;font-weight:600;color:#1A5276">
            {"For a more accurate result, go to the ADD MEASUREMENTS tab above and enter your body measurements." if L=="en" else "Ongeraho ibiro n'uburebure kugirango ibisubizo byawe bibe byiza kurushaho."}
          </div>
        </div>
        <div style="font-size:11px;color:var(--text3);margin-top:4px;margin-bottom:8px">
          {"If you change any answer above, click Calculate My Risk again to update your result." if L=="en" else "Niba wahinduje igisubizo, kanda Bara Ibyago Byange nanone kugira ngo ubone ibisubizo bishya."}
        </div>""", unsafe_allow_html=True)

# ════ LAYER 2 ════
with tab2:
    if not st.session_state['assessed']:
        st.info(T['add_l1'])
    else:
        st.caption(T['l2'])
        st.markdown(f'<div class="slbl">{T["body"]}</div>', unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        # ── LEFT COLUMN — Weight + Height → BMI ──────────────────────────────
        with col_left:
            wt = st.number_input(T['wt'], 30.0, 150.0, 65.0, 0.5, key='k_wt')
            ht = st.number_input(T['ht'], 130.0, 210.0, 165.0, 0.5, key='k_ht')
            bmi_c = round(wt / ((ht/100)**2), 1)
            if bmi_c >= 30:   bmi_tag = f'<span class="abadge abadge-alert">{T["bmi_obese"]}</span>'
            elif bmi_c >= 25: bmi_tag = f'<span class="abadge abadge-warn">{T["bmi_over"]}</span>'
            else:             bmi_tag = f'<span class="abadge abadge-ok">{T["bmi_normal"]}</span>'
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:14px;margin-top:8px">
              <div style="font-size:28px;font-weight:700;color:var(--navy)">{bmi_c}</div>
              <div style="font-size:10px;color:var(--text3);text-transform:uppercase;
                          letter-spacing:1px;margin:4px 0">BMI kg/m²</div>
              {bmi_tag}
            </div>""", unsafe_allow_html=True)

        # ── RIGHT COLUMN — Waist (optional) ──────────────────────────────────
        with col_right:
            wst_measured = st.checkbox(T['wst_check'], key='k_wst_check')
            if wst_measured:
                wst_input = st.number_input(T['wst'], 0.0, 150.0, 0.0, 0.5, key='k_wst')
                st.caption(T['wst_help'])
                wst = float(wst_input) if wst_input > 0 else np.nan
                if wst_input == 0:
                    st.warning('Please enter your waist measurement.' if L=='en'
                               else "Injiza ingano y'ikibuno.")
                else:
                    cutoff  = 102 if st.session_state['pd_data'].get('sex',2)==1 else 88
                    wst_tag = f'<span class="abadge abadge-alert">{T["wst_high"]}</span>' \
                              if wst > cutoff else \
                              f'<span class="abadge abadge-ok">{T["wst_normal"]}</span>'
                    st.markdown(f"""
                    <div class="card" style="text-align:center;padding:14px;margin-top:8px">
                      <div style="font-size:28px;font-weight:700;color:var(--navy)">{wst:.0f}</div>
                      <div style="font-size:10px;color:var(--text3);text-transform:uppercase;
                                  letter-spacing:1px;margin:4px 0">Waist cm</div>
                      {wst_tag}
                    </div>""", unsafe_allow_html=True)
            else:
                wst = np.nan

        if st.button(T['btn2'], key='btn2'):
            vals = dict(st.session_state['pd_data'])
            vals.update({'bmi': bmi_c,
                         'm14': float(wst) if wst_measured and not np.isnan(wst) else np.nan})
            prob = predict(vals, USE_XGB)
            st.session_state.update({'prob': prob, 'layers': 2, 'pd_data': vals})

        if st.session_state['layers'] >= 2:
            show_result(st.session_state['prob'], THRESHOLD, 2, USE_XGB,
                        st.session_state['pd_data'], L, T)

# ── DISCLAIMER ────────────────────────────────────────────────────────────────
st.markdown(f'<div class="disc">{T["disc"]}</div>',
            unsafe_allow_html=True)
