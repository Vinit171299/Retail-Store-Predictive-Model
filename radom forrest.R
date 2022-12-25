library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(vip)
library(rpart.plot)
library(DALEXtra)

install.packages("ranger")

path1<-"C:\\Users\\xyz\\Desktop\\vinit\\R project Data\\"
path1
store_train<-read.csv(paste0(path1,"store_train.csv"),sep = ',',stringsAsFactors = FALSE)

store_test<-read.csv(paste0(path1,"store_test.csv"),sep = ',',stringsAsFactors = FALSE)

glimpse(store_train)
glimpse(store_test)


dp_pipe=recipe(store~.,data=store_train) %>% 
  update_role(Id,storecode,countytownname,countyname,Areaname,new_role = "drop_vars") %>%
  update_role(store_Type,state_alpha,new_role="to_dummies") %>% 
  step_rm(has_role("drop_vars")) %>% 
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>%
  step_impute_median(all_numeric(),-all_outcomes())

dp_pipe=prep(dp_pipe)

train=bake(dp_pipe,new_data=NULL)
test=bake(dp_pipe,new_data=store_test)

head(train)
train$store <- as.factor(train$store)

rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

folds = vfold_cv(train, v = 5)

rf_grid = grid_regular(mtry(c(5,25)), trees(c(100,500)),
                       min_n(c(2,10)),levels = 3)

my_res=tune_grid(
  rf_model,
  store~.,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
)

autoplot(my_res)+theme_light()

my_res %>% show_best()

final_rf_fit=rf_model %>% 
  set_engine("ranger",importance='permutation') %>% 
  finalize_model(select_best(my_res,"roc_auc")) %>% 
  fit(store~.,data=train)

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

train_pred=predict(final_rf_fit,new_data = train,type="prob") %>% select(.pred_1)

test_pred=predict(final_rf_fit,new_data = test,type="prob") %>% select(.pred_1)

train.score=train_pred$.pred_1

real=train$store


test_pred=predict(final_rf_fit,new_data = test,type="prob") %>% select(.pred_1)
write.csv(test_pred,'Vinit_Pawar_P2_part2.csv',row.names = F)


rocit = ROCit::rocit(score = train.score, 
                     class = real) 

kplot=ROCit::ksplot(rocit)

my_cutoff=kplot$`KS Cutoff`

test_hard_class=as.numeric(test_pred>my_cutoff)

