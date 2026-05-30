import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================
# 1. PARAMETRE DU MÅ FYLLE INN SELV
# ============================================

excel_path = r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/2026_02_Diffusion/MRIScanData/H2_N2_110_diff45V/H2_N2_diffusion_results.xlsx"


# Kammer-volum (i m^3)
V_A = 7.06858e-5   # f.eks. 1 liter = 1e-3 m^3
V_B = 7.06858e-5

# Geometri av ventil (m)
A = 2.82743e-5     # tverrsnitt i m^2
L = 0.201     # lengde i m


import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

files = [
    r"path/exp1.xlsx",
    r"path/exp2.xlsx",
    r"path/exp3.xlsx",
    r"path/exp4.xlsx",
    r"path/exp5.xlsx"
]

results = []

for f in files:
    df = pd.read_excel(f)
    t = df["Time_seconds"].values
    S_A = df["ROI1_mean"].values

    def exp_model(t, S_eq, S0, tau):
        return S_eq + (S0 - S_eq) * np.exp(-t / tau)

    popt, _ = curve_fit(exp_model, t, S_A, p0=[S_A[-1], S_A[0], (t[-1]-t[0])/5])
    S_eq, S0, tau = popt
    k = 1 / (tau * (1/V_A + 1/V_B))
    D = k * L / A

    results.append((f.split("/")[-1], tau, k, D))

df_res = pd.DataFrame(results, columns=["Experiment", "tau (s)", "k (1/s)", "D_eff (m²/s)"])
df_res.loc["Mean"] = ["Mean", df_res["tau (s)"].mean(), df_res["k (1/s)"].mean(), df_res["D_eff (m²/s)"].mean()]
df_res.loc["Std"] = ["Std", df_res["tau (s)"].std(), df_res["k (1/s)"].std(), df_res["D_eff (m²/s)"].std()]
print(df_res)

# ============================================
# 2. LES DATA
# ============================================

df = pd.read_excel(excel_path)

t = df["Time_seconds"].values
S_A = df["ROI1_mean"].values   # kammer med H2
S_B = df["ROI2_mean"].values   # kammer med N2

# ============================================
# 3. EKSPONENSIAL FITT AV SIGNAL
# ============================================

def exp_model(t, S_eq, S0, tau):
    return S_eq + (S0 - S_eq) * np.exp(-t / tau)

# Gjett startverdier
S_eq_guess = S_A[-1]
S0_guess = S_A[0]
tau_guess = (t[-1] - t[0]) / 5

popt, pcov = curve_fit(exp_model, t, S_A, p0=[S_eq_guess, S0_guess, tau_guess])
S_eq_fit, S0_fit, tau_fit = popt

print("\n==============================")
print("FITTRESULTATER")
print("==============================")
print(f"Tidskonstant τ = {tau_fit:.4f} s")
print(f"S_eq = {S_eq_fit:.4f}")
print(f"S0 = {S0_fit:.4f}")

# ============================================
# 4. BEREGN k FRA TAU
# ============================================

k = 1 / (tau_fit * (1/V_A + 1/V_B))

print("\n==============================")
print("TRANSPORTKONSTANT k")
print("==============================")
print(f"k = {k:.6e}  [1/s]")

# ============================================
# 5. BEREGN DIFFUSJONSKOEFFISIENT D
# ============================================

D = k * L / A

print("\n==============================")
print("DIFFUSJONSKOEFFISIENT D")
print("==============================")
print(f"D = {D:.6e}  m^2/s")

# ============================================
# 6. PLOTT RESULTAT
# ============================================

t_fit = np.linspace(t.min(), t.max(), 500)
S_fit = exp_model(t_fit, S_eq_fit, S0_fit, tau_fit)

plt.figure(figsize=(10,5))
plt.plot(t, S_A, 'o', label="Målt S_A(t)")
plt.plot(t_fit, S_fit, '-', label="Eksponential fit")
plt.xlabel("Tid (s)")
plt.ylabel("Signalintensitet (ROI1_mean)")
plt.title("Diffusjonsanalyse – MR signal i kammer A")
plt.legend()
plt.grid(True)
plt.show()