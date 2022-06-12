import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image

@st.cache
def access_data(selected_option): 
    files = {"Lagoa dos Patos/RS":"Lagoa_dos_Patos.csv",
            "Pontal do Paraná/PR":"Pontal_do_Parana.csv",
            "Praia do Cassino/RS":"Praia_do_Cassino.csv",
            "Praia do Forte/BA":"Praia_do_Forte.csv",
            "Recife/PE":"Recife.csv",
            "Rio de Janeiro/RS":"Rio_de_Janeiro_-_2.csv",
            "Santos/SP":"Santos.csv",
            "Tramandai/RS":"Tramandai.csv"}

    path = "https://raw.githubusercontent.com/guerchen/WaveEnergy/main/"+files[selected_option]
    return pd.read_csv(path)

def format_date(dataframe_to_format):
    dataframe = dataframe_to_format.copy()
    dataframe["data"] = dataframe["yyyy"].map(str)+"-"+dataframe[" mm"].map(str)+"-"+dataframe[" dd"].map(str)+" "+dataframe[" hour"].map(str)+":"+dataframe[" min"].map(str)+":"+dataframe[" seg"].map(str)
    dataframe["data"] = pd.to_datetime(dataframe["data"])
    return dataframe

def energy_output_model(T, H):
    return -2.101*(T**2) + 10.295*(H**2) + 0.973*T*H + 32.260*T + 44.291*H - 73.098

def wake_effect(x,y,T,H):
    a0 = 5.0260*(T**2) - 2.4044*(H**2) + 1.1170*T*H - 73.4951*T + 0.7381*H + 265.008
    b0 = -0.0026*(T**2) - 0.0021*(H**2) + 0.0024*T*H + 0.0269*T - 0.0156*H - 0.3543
    c0 = 0.3484*(T**2) - 0.4561*(H**2) + 0.3576*T*H - 3.3016*T - 1.8723*H + 13.1247
    d0 = 0.0006*(T**2) + 0.0101*(H**2) - 0.0078*T*H + 0.0347*T + 0.0456*H - 0.0586
    e0 = 0.6170*(T**2) + 0.0070*(H**2) - 0.6494*T*H - 5.5661*T + 5.3735*H - 23.232
    f0 = -0.0113*(T**2) - 0.0063*(H**2) + 0.0060*T*H + 0.1908*T - 0.0314*H - 0.5914
    g0 = -0.5522*(T**2) + 3.7169*(H**2) - 0.8187*T*H + 27.789*T - 30.309*H - 97.966
    h0 = 0.0041*(T**2) - 0.0013*(H**2) + 0.0030*T*H - 0.1233*T - 0.0440*H + 0.2728
    i0 = 0.8990*(T**2) + 0.7541*(H**2) - 0.0275*T*H + 2.1218*T - 1.6382*H - 6.2047
    j0 = -0.0056*(T**2) - 0.0004*(H**2) + 0.0044*T*H + 0.0724*T - 0.0216*H + 0.1748
    k0 = 1.1515*(T**2) + 0.2448*(H**2) + 0.3270*T*H - 17.7210*T - 2.8051*H + 73.3434
    l0 = -0.0048*(T**2) - 0.0080*(H**2) - 0.0020*T*H + 0.0931*T + 0.0353*H - 0.3026
    
    delta_H_d = ((x+a0)**b0)*(np.exp(-((y-c0-d0*x)**2)/((e0+f0*x)**2))+np.exp(-((y+c0+d0*x)**2)/((e0+f0*x)**2)))
    delta_H_s = ((x+g0)**h0)*(np.exp(-((y-i0-j0*x)**2)/((k0+l0*x)**2))+np.exp(-((y+i0+j0*x)**2)/((k0+l0*x)**2)))
    
    return delta_H_d + delta_H_s

st.title("Study of potential oscillating wave surge energy generation at selected beaches in Brazil")

st.subheader('Created by: Ariel Guerchenzon')

st.markdown("This website was created to facilitate data visualization regarding potetial energy generation using oscillating wave surge converters along the coast of Brazil. The data was extracted from GOOS-Brazil (Global Ocean Observating System) [1]. The equations for potential energy generation and wake effect behind the generator were based on Y. Wang and Z. Liu's [2] study: Proposal of novel analytical wake model and GPU-accelerated array optimization method for oscillating wave surge energy converter.")

df_praias = pd.DataFrame(["Lagoa dos Patos/RS","Pontal do Paraná/PR","Praia do Cassino/RS","Praia do Forte/BA","Recife/PE","Rio de Janeiro/RS","Santos/SP","Tramandai/RS"])

option = st.selectbox('Select beach:',df_praias)

wave_data = access_data(option)

st.header("Visualizing the raw data:")

st.dataframe(wave_data)

wave_data = format_date(wave_data)

y1 = wave_data[" altura"]
y2 = wave_data[" periodo"]
x = wave_data["data"]

# plot
fig, ax = plt.subplots(2,1)

ax[0].scatter(x, y1, s=2.0)
ax[0].set_ylabel("Wave height (m)")
ax[0].set_xticks([])

tick_interval = 800 # Defining tick interval in the x axis to make label readable 
ticks_x_axis = [wave_data["data"].iloc[i*tick_interval] for i in range(wave_data["data"].shape[0]//tick_interval + 1)]

ax[1].scatter(x, y2, s=2.0)
ax[1].set_xlabel("Date")
ax[1].set_ylabel("Wave period (s)")
ax[1].set_xticks(ticks=ticks_x_axis)

ax[0].set_title("Wave profile at "+option+", according to GOOS-Brasil")

st.header("Scatter plot of raw data:")

st.pyplot(fig)

st.header("Energy-output model:")

st.markdown("The following equation was obtained from [2] and provides a numeric model for instantaneous energy output from an oscillating wave surge converter. 'H' represents wave height in meters and 'T' represents wave period in seconds.")

st.image("https://raw.githubusercontent.com/guerchen/WaveEnergy/main/Power.png")

st.markdown(f"Below is a table of statistical metrics from the instantaneous energy output model for {option}.")

df_potencia = pd.DataFrame()
df_potencia = energy_output_model(wave_data[" periodo"],wave_data[" altura"])

st.dataframe(df_potencia.describe())

st.header("Wake effect model")

st.markdown("[2] also provides a numerical model for calculating the wake effect behind an oscillating wave surge converter. This model is especially usefull for designing optimized generator grids. Below, the equation is displayed.")

st.image("https://raw.githubusercontent.com/guerchen/WaveEnergy/main/Wakemodel.png")

st.markdown("The equation coefficients are as follows:")

st.image("https://raw.githubusercontent.com/guerchen/WaveEnergy/main/coef.png")

st.markdown(f"Using median wave height and period from {option}, its possible to simulate the resulting wake effect map.")

T = wave_data[" periodo"].median()
H = wave_data[" altura"].median()

y = np.arange(0, 100, 1)
x = np.arange(0, 200, 1)
Z = np.zeros((x.shape[0],y.shape[0]))

for i in range(200):
    for j in range(100):
        Z[i][j] = wake_effect(i,j,T,H)

fig2, ax = plt.subplots()
ax.pcolormesh(y, x, Z)
st.pyplot(fig2)

st.markdown("The picture above represents an oscillating wave surge converter placed at the origin and wave direction going up-right. The darker areas represent areas unaffected by wake effect and are, therefore, more suitable for the placement of other generators. Using this data it's possible to design a power optimized generator grid.")

st.header("References:")

st.markdown("[1] - GOOS Brasil. Rede de ondas do litoral brasileiro. Available at: <http://www.goosbrasil.org/rede_ondas/dados/>. Access in 21/05/2022 at 4:33 PM.")
st.markdown("[2] - WANG, Yize; LIU, Zhenqing. Proposal of novel analytical wake model and GPU-accelerated array optimization method for oscillating wave surge energy converter. Renewable Energy, 2021, 179: 563-583.")

st.header("Code used for simulations")

code ="""def energy_output_model(T, H):
    return -2.101*(T**2) + 10.295*(H**2) + 0.973*T*H + 32.260*T + 44.291*H - 73.098

def wake_effect(x,y,T,H):
    a0 = 5.0260*(T**2) - 2.4044*(H**2) + 1.1170*T*H - 73.4951*T + 0.7381*H + 265.008
    b0 = -0.0026*(T**2) - 0.0021*(H**2) + 0.0024*T*H + 0.0269*T - 0.0156*H - 0.3543
    c0 = 0.3484*(T**2) - 0.4561*(H**2) + 0.3576*T*H - 3.3016*T - 1.8723*H + 13.1247
    d0 = 0.0006*(T**2) + 0.0101*(H**2) - 0.0078*T*H + 0.0347*T + 0.0456*H - 0.0586
    e0 = 0.6170*(T**2) + 0.0070*(H**2) - 0.6494*T*H - 5.5661*T + 5.3735*H - 23.232
    f0 = -0.0113*(T**2) - 0.0063*(H**2) + 0.0060*T*H + 0.1908*T - 0.0314*H - 0.5914
    g0 = -0.5522*(T**2) + 3.7169*(H**2) - 0.8187*T*H + 27.789*T - 30.309*H - 97.966
    h0 = 0.0041*(T**2) - 0.0013*(H**2) + 0.0030*T*H - 0.1233*T - 0.0440*H + 0.2728
    i0 = 0.8990*(T**2) + 0.7541*(H**2) - 0.0275*T*H + 2.1218*T - 1.6382*H - 6.2047
    j0 = -0.0056*(T**2) - 0.0004*(H**2) + 0.0044*T*H + 0.0724*T - 0.0216*H + 0.1748
    k0 = 1.1515*(T**2) + 0.2448*(H**2) + 0.3270*T*H - 17.7210*T - 2.8051*H + 73.3434
    l0 = -0.0048*(T**2) - 0.0080*(H**2) - 0.0020*T*H + 0.0931*T + 0.0353*H - 0.3026
    
    delta_H_d = ((x+a0)**b0)*(np.exp(-((y-c0-d0*x)**2)/((e0+f0*x)**2))+np.exp(-((y+c0+d0*x)**2)/((e0+f0*x)**2)))
    delta_H_s = ((x+g0)**h0)*(np.exp(-((y-i0-j0*x)**2)/((k0+l0*x)**2))+np.exp(-((y+i0+j0*x)**2)/((k0+l0*x)**2)))
    
    return delta_H_d + delta_H_s
"""
st.code(code, language='python')