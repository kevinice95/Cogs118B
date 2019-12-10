df = read.csv(file.choose())
head(df, 5)
colnames(df)
head(dplyr::select(df,priors_count,priors_count.1))
recid = dplyr::select(df,is_recid,two_year_recid)
sum(recid[,1]!=recid[,2])
library(dplyr)
df.filt=dplyr::select(df,sex,age_cat,race,juv_fel_count,juv_misd_count,priors_count,c_charge_degree,type_of_assessment,decile_score.1,v_type_of_assessment,v_decile_score,is_recid,is_violent_recid)
head(df.filt)
colnames(df.filt)[colnames(df.filt)=="decile_score.1"]="r_score"
colnames(df.filt)[colnames(df.filt)=="v_decile_score"]="v_score"
df.filt=df.filt[,!grepl("assessment",colnames(df.filt))]
head(df.filt)
levels(df.filt$sex)
df.filt$sex=relevel(df.filt$sex,"Male")
levels(df.filt$age_cat)
df.filt$age_cat=relevel(df.filt$age_cat,"Less than 25")
levels(df.filt$race)
df.filt$race=relevel(df.filt$race,"Caucasian")
levels(df.filt$c_charge_degree)
df.filt$c_charge_degree=relevel(df.filt$c_charge_degree,"M")

df.num=df.filt
df.num$sex=as.numeric(df.num$sex)-1
df.num$age_cat=as.numeric(df.num$age_cat)-1
df.num$c_charge_degree=as.numeric(df.num$c_charge_degree)-1
df.num$race  = factor(c(df.num$race))
head(df.num)

library(factoextra)
df.pca <- prcomp(data.matrix(df.num), scale = TRUE)
fviz_pca_var(df.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)

