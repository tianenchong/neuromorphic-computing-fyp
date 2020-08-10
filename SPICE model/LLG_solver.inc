************************************************************************************
************************************************************************************
** Title:  LLG_solver.inc
** Author: Jongyeon Kim, VLSI Research Lab @ UMN
** Email:  kimx2889@umn.edu
************************************************************************************
** At the given MTJ dimensions and material parameters, the dynamic tr motion is
** implemented by LLG equation according to the type of MTJ. 
************************************************************************************

.subckt LLG Mx My Mz Is Istt Ias Tmp thi lx='65n' ly='65n' lz='1.8n'  lxshm ='70n' lyshm= '40n' lzshm= '2.2n' Ms0='1075' P0='0.715' alpha='0.01' MA='1' ini='0' Kp='1.2e7' SHA='0.4' H_app = '400'  UI = '0' InA = ' 0.01'  x_ad = '0'

.include pwlFile_92.in
.param thermal = '1'

*** Physical parameters **************************
.param pi='355/113'
.param gamma='2.8e6*2*pi'
.param h='6.625e-27/(2*pi)'
.param e='1.602e-19'
.param kb='1.38e-16'
.param lambda_sf = '3.5e-9'


*** Temp. dependent parameters *******************
.param Tcurie='1420'
.param beta='0.4'
.param asp='2e-5'
E_Ms Ms 0 vol='Ms0*(1-v(Tmp)/Tcurie)^beta'
E_P  P  0 vol='P0*(1-asp*v(Tmp)^1.5)'
*E_Ms Ms 0 vol='Ms0'
*E_P  P  0 vol='P0'

*** Magnetization of pinned layer ****************
.param Mpx='0.0'
.param Mpy='1-MA'
.param Mpz='MA'

.param Mpxshm='0.0'
.param Mpyshm='1.0'
.param Mpzshm='0.0'

*** Shape anisotropy - Demagnetizing factors *****
.param Nsh(a,b,c)='1/pi*((b^2-c^2)/(2*b*c)*log((sqrt(a^2+b^2+c^2)-a)/(sqrt(a^2+b^2+c^2)+a))+(a^2-c^2)/(2*a*c)*log((sqrt(a^2+b^2+c^2)-b)/(sqrt(a^2+b^2+c^2)+b))+b/(2*c)*log((sqrt(a^2+b^2)+a)/(sqrt(a^2+b^2)-a))+a/(2*c)*log((sqrt(a^2+b^2)+b)/(sqrt(a^2+b^2)-b))+c/(2*a)*log((sqrt(b^2+c^2)-b)/(sqrt(b^2+c^2)+b))+c/(2*b)*log((sqrt(a^2+c^2)-a)/(sqrt(a^2+c^2)+a))+2*atan((a*b)/(c*sqrt(a^2+b^2+c^2)))+(a^3+b^3-2*c^3)/(3*a*b*c)+(a^2+b^2-2*c^2)/(3*a*b*c)*sqrt(a^2+b^2+c^2)+c/(a*b)*(sqrt(a^2+c^2)+sqrt(b^2+c^2))-((b^2+a^2)^(3/2)+(b^2+c^2)^(3/2)+(a^2+c^2)^(3/2))/(3*a*b*c))'

.param Nx='4*pi*Nsh(lz,ly,lx)'
.param Ny='4*pi*Nsh(lz,lx,ly)'
.param Nz='4*pi*Nsh(ly,lx,lz)'


****Test****
E_NX_1 NX_1 0 vol = 'Nx'
E_NX_2 NX_2 0 vol = 'Ny'
E_NX_3 NX_3 0 vol = 'Nz'
*** Initialization ********************************
.param Msi='Ms0*(1-Tmp0/Tcurie)^beta'
*.param Msi='Ms0'
.param cPMA='(((Kp)-0.5*(Nz)*(Msi)*(Msi))*lx*ly*lz*1e6)/(kb*(Tmp0))'

.param thi='asin((1/(2*cPMA))^(1/2))*(1-UI) + UI*InA'
E_thi thi 0 vol='asin((1/(2*cPMA))^(1/2))*(1-UI) + UI*InA'
*.print delta=par('cPMA')
E_tsf delta_1_tsf 0 vol = 'cPMA'

.param Mx0='0.0'
.param My0='sin(thi)'
.param Mz0='(1-2*ini)*cos(thi)'


.ic v(Mx)='Mx0'
.ic v(My)='My0'
.ic v(Mz)='Mz0'


*** Thermal fluctuation field *********************

*** Demagnetizating field for in-plane MTJ ********
E_Hdx Hdx 0 vol='-Nx*v(Mx)*v(Ms)'
E_Hdy Hdy 0 vol='-Ny*v(My)*v(Ms)'
E_Hdz Hdz 0 vol='-Nz*v(Mz)*v(Ms)'

E_Hiz Hiz 0 vol='2*Kp/v(Ms)*v(Mz)'

*** applied field *****

E_Happlyx Happliedx 0 vol = 'H_app'
E_Happlyy Happliedy 0 vol = '00'
E_Happlyz Happliedz 0 vol = '0.0'

*** Effective anisotropy field ********************
E_Hefx Hefx 0 vol='v(Hdx)+v(Happliedx)+1*thermal*v(Hthx)' 
E_Hefy Hefy 0 vol='v(Hdy)+v(Happliedy)+1*thermal*v(Hthy)' 
E_Hefz Hefz 0 vol='v(Hdz)+v(Happliedz)+MA*v(Hiz)+thermal*v(Hthz)'

*** Field magnitude from SHE  **********

R_Is Is 0 '1*h/(2*e*lyshm*lzshm*lz*1e6*v(Ms))*1*SHA *(1-(1/cosh(lzshm/lambda_sf)))'


*E_IS_test Is_test 0 vol ='1*SHA*h/(2*e*lx*ly*lz*1e6*v(Ms))*(1-(1/cosh(lzshm/lambda_sf)) )'

E_sha_inst sha_inst 0 vol = 'SHA*(1-(1/cosh(lzshm/lambda_sf) ) )'
E_pshe pshe 0 vol = 'V(sha_inst)*(lx*ly/lyshm/lzshm)'
E_xad xad 0 vol ='x_ad'



*** Polarized spin current (J=Ias/lx/ly) **********
R_Is2 Istt 0 '4*V(Ias)*v(P)*h/(2*e*pi*lx*ly*lz*1e6*v(Ms))'
E_IS_test2 Is2_test 0 vol ='V(Ias)*v(P)*4*h/(2*e*pi*lx*ly*lz*1e6*v(Ms))'

*** LLG solving for Mx, My, Mz ********************

C_Mx Mx 0 '(1+alpha*alpha)/gamma'
G_dMx_prec 0 Mx cur='(-(v(My)*v(Hefz)-v(Hefy)*v(Mz)))'
G_dMx_damp 0 Mx cur='-alpha*(v(My)*(v(Mx)*v(Hefy)-v(Hefx)*v(My))-(v(Mz)*v(Hefx)-v(Hefz)*v(Mx))*v(Mz))'
G_dMx_torq 0 Mx cur='v(Istt)*(v(My)*(v(Mx)*Mpy-Mpx*v(My))-(v(Mz)*Mpx-Mpz*v(Mx))*v(Mz))'
G_dMx_torq2 0 Mx cur='v(Is)*(v(My)*(v(Mx)*Mpyshm-Mpxshm*v(My))-(v(Mz)*Mpxshm-Mpzshm*v(Mx))*v(Mz))'
*G_dMx_antidamp 0 Mx cur = 'v(Is)*x_ad*(v(My)*(v(Mx)*v(Hefy)-v(Hefx)*v(My))-(v(Mz)*v(Hefx)-v(Hefz)*v(Mx))*v(Mz))''
*G_dMx_antidamp 0 Mx cur = 'v(Is)*x_ad*(v(My)*(v(Mx)*Mpyshm-Mpxshm*v(My))-(v(Mz)*Mpxshm-Mpzshm*v(Mx))*v(Mz))''

*G_dMx_antidamp1 0 Mx cur = 'v(Istt)*x_ad*(v(My)*(v(Mx)*Mpy-Mpx*v(My))-(v(Mz)*Mpx-Mpz*v(Mx))*v(Mz))''

G_dMx_antidamp 0 Mx cur='(v(Is)*x_ad*(v(My)*Mpzshm-Mpyshm*v(Mz)))'
G_dMx_antidamp1 0 Mx cur='(v(Istt)*x_ad*(v(My)*Mpz-Mpy*v(Mz)))'



C_My My 0 '(1+alpha*alpha)/gamma'
G_dMy_prec 0 My cur='(-(v(Mz)*v(Hefx)-v(Hefz)*v(Mx)))'
G_dMy_damp 0 My cur='-alpha*(v(Mz)*(v(My)*v(Hefz)-v(Hefy)*v(Mz))-(v(Mx)*v(Hefy)-v(Hefx)*v(My))*v(Mx))'
G_dMy_torq 0 My cur='v(Istt)*(v(Mz)*(v(My)*Mpz-Mpy*v(Mz))-(v(Mx)*Mpy-Mpx*v(My))*v(Mx))'
G_dMy_torq2 0 My cur='v(Is)*(v(Mz)*(v(My)*Mpzshm-Mpyshm*v(Mz))-(v(Mx)*Mpyshm-Mpxshm*v(My))*v(Mx))'
*G_dMy_antidamp 0 My cur='v(Is)*x_ad*(v(Mz)*(v(My)*v(Hefz)-v(Hefy)*v(Mz))-(v(Mx)*v(Hefy)-v(Hefx)*v(My))*v(Mx))'
*G_dMy_antidamp 0 My cur='v(Is)*x_ad*(v(Mz)*(v(My)*Mpzshm-Mpyshm*v(Mz))-(v(Mx)*Mpyshm-Mpxshm*v(My))*v(Mx))'
*G_dMy_antidamp1 0 My cur='v(Istt)*x_ad*(v(Mz)*(v(My)*Mpz-Mpy*v(Mz))-(v(Mx)*Mpy-Mpx*v(My))*v(Mx))'
G_dMy_antidamp 0 My cur='(v(Is)*x_ad*(v(Mz)*Mpxshm-Mpzshm*v(Mx)))'
G_dMy_antidamp1 0 My cur='(v(Istt)*x_ad*(v(Mz)*Mpx-Mpz*v(Mx)))'


C_Mz Mz 0 '(1+alpha*alpha)/gamma'
G_dMz_prec 0 Mz cur='-(v(Mx)*v(Hefy)-v(Hefx)*v(My))'
G_dMz_damp 0 Mz cur='-alpha*(v(Mx)*(v(Mz1)*v(Hefx)-v(Hefz)*v(Mx))-(v(My)*v(Hefz)-v(Hefy)*v(Mz1))*v(My))'
G_dMz_torq 0 Mz cur='v(Istt)*(v(Mx)*(v(Mz1)*Mpx-Mpz*v(Mx))-(v(My)*Mpz-Mpy*v(Mz1))*v(My))'
G_dMz_torq2 0 Mz cur='v(Is)*(v(Mx)*(v(Mz1)*Mpxshm-Mpzshm*v(Mx))-(v(My)*Mpzshm-Mpyshm*v(Mz1))*v(My))'
E_Mz1 Mz1 0 vol='v(Mz)' max='cos(v(thi))' min='-cos(v(thi))'
*G_dMz_antidamp 0 Mz cur='v(Is)*x_ad*(v(Mx)*(v(Mz1)*v(Hefx)-v(Hefz)*v(Mx))-(v(My)*v(Hefz)-v(Hefy)*v(Mz1))*v(My))'
*G_dMz_antidamp 0 Mz cur='v(Is)*x_ad*(v(Mx)*(v(Mz1)*Mpxshm-Mpzshm*v(Mx))-(v(My)*Mpzshm-Mpyshm*v(Mz1))*v(My))'
*G_dMz_antidamp1 0 Mz cur='v(Istt)*x_ad*(v(Mx)*(v(Mz1)*Mpx-Mpz*v(Mx))-(v(My)*Mpz-Mpy*v(Mz1))*v(My))'
G_dMz_antidamp 0 Mz cur='v(Is)*x_ad*(v(Mx)*Mpyshm-Mpxshm*v(My))'
G_dMz_antidamp1 0 Mz cur='v(Istt)*x_ad*(v(Mx)*Mpy-Mpx*v(My))'
.ends

