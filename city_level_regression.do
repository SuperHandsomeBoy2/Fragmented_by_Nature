clear
set more off
*capture ssc install reghdfe
*capture ssc install ivreghdfe
*capture ssc install outreg2
*capture ssc install asdoc
*capture ssc install estout
cd "/Users/saiz/Desktop/Dropbox (MIT)/RESEARCH/BOSTONGEOGRAPHY/WORLD/regressions_DETOUR/"
use "DATAGEN/URBAN_BARRIERS.dta"


/* DETOUR IS MOSTLY EXPLAINED BY GEO BARRIERS */
xi: reg detour share_barrier_10km, 
est store T1_1
xi: reg detour  nonconcavity_10km
est store T1_2
xi: reghdfe detour share_barrier_10km coastal capital dry log_precip log_temp logdpcap15 lopo15 , absorb(ctr_mn_iso soil climate biome)
est store T1_3
xi: reghdfe detour nonconcavity_10km coastal capital dry log_precip log_temp logdpcap15 lopo15 , absorb(ctr_mn_iso soil climate biome)
est store T1_4
xi: reghdfe detour share_barrier_10km nonconcavity_10km coastal capital dry log_precip log_temp logdpcap15 lopo15 , absorb(ctr_mn_iso soil climate biome)
est store T1_5

codebook ctr_mn_iso // 182 countries
codebook ucid // 13,135 unique cities 

bys ctr_mn_iso: gen n = _N // how many cities in each country
tab n
tab ctr_mn_iso if n == 1 // which countries only have one city, these countries will be dropped after controlling for country fixed effects


********************************
xi: reghdfe lopo15 nonconcavity_10km  coastal capital dry log_precip log_temp  , absorb(ctr_mn_iso soil climate biome)
esttab, beta not
est store T2_1
xi: reghdfe logdpcap15  nonconcavity_10km  coastal capital dry log_precip log_temp lopo15 , absorb(ctr_mn_iso soil climate biome)
esttab, beta not
est store T2_2
xi: reghdfe lolight  nonconcavity_10km  coastal capital dry log_precip log_temp lopo15 , absorb(ctr_mn_iso soil climate biome)
esttab, beta not
est store T2_3

xi: reghdfe lodensb nonconcavity_10km coastal capital dry log_precip log_temp logdpcap15 lopo15 , absorb(ctr_mn_iso soil climate biome)
esttab, beta not
est store T2_4
xi: reghdfe sharebu nonconcavity_10km  coastal capital dry log_precip log_temp logdpcap15 lopo15 , absorb(ctr_mn_iso soil climate biome)
esttab, beta not
est store T2_5
xi: reghdfe log_height nonconcavity_10km  coastal capital dry log_precip log_temp logdpcap15 lopo15 , absorb(ctr_mn_iso soil climate biome)
esttab, beta not
est store T2_6
xi: reghdfe landinefficiency nonconcavity_10km  coastal capital dry log_precip log_temp logdpcap15 lopo15 , absorb(ctr_mn_iso soil climate biome)
esttab, beta not
est store T2_7

xi: reghdfe e_gr_av14  nonconcavity_10km  coastal capital dry log_precip log_temp  logdpcap15 lopo15,   absorb(ctr_mn_iso soil climate biome)
esttab, beta not
est store T2_8
xi: reghdfe loco2r  nonconcavity_10km  coastal capital dry log_precip log_temp  logdpcap15 lopo15,   absorb(ctr_mn_iso soil climate biome)
esttab, beta not
est store T2_9
xi: reghdfe loco2t nonconcavity_10km  coastal capital dry log_precip log_temp  logdpcap15 lopo15,   absorb(ctr_mn_iso soil climate biome)
esttab, beta not
est store T2_10

outreg2 [T2_1 T2_2 T2_3 T2_4 T2_5 T2_6 T2_7 T2_8 T2_9 T2_10] using Table2.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
 
 
asdoc sum detour share_barrier_10km nonconcavity_10km  coastal capital dry log_precip log_temp lopo15 logdpcap15  lolight ///
lodensb sharebu log_height landinefficiency loco2r loco2t, replace word label

/* INTERACTIONS WITH "DEMAND" PROXIED BY NATIONAL GROWTH */

gen iso=ctr_mn_iso
sort iso
merge iso using "DATAGEN/Country_level/countryLevelInicators.dta"


gen nonconcavity_10km_UrbanPopGrowth=UrbanPopGrowth*nonconcavity_10km

/* annuitized growth of city */
gen urbgro=(p15/p00)^(1/15)-1
gen lop00=log(p00)
xi: reghdfe urbgro nonconcavity_10km nonconcavity_10km_UrbanPopGrowth UrbanPopGrowth coastal capital dry log_precip log_temp  , absorb(ctr_mn_iso soil climate biome)

/*
 




xi: reg sdg_a2g14 share_barrier_10kmfix detour log_elev_mean log_elev_std log_grade_mean log_intersect_count log_k_avg log_length_mean prop_4way prop_3way prop_deadend self_loop_proportion sdg_os15mx logdpcap15 lopo15 pcflood i.world_region if detour>=0  
  est store N6_1
  outreg2[N6_1] using TableN6.xls, stats(coef,se) addstat(Ajusted R2,`e(r2_a)') replace word label
