#------------------------------------------EDA---------------------------
summary(df)

#-------------Univariate analysis per data type------------------
df %>%
select_if(is.factor) %>% 
    gather() %>% 
    ggplot(aes(x = value)) +
    facet_wrap(~ key, scales = "free") +
    geom_bar()

df %>%
    select_if(is.logical) %>% 
    gather() %>% 
    ggplot(aes(x = value)) +
    facet_wrap(~ key, scales = "free") +
    geom_bar()


df %>%
    select_if(is.numeric) %>% 
    gather() %>% 
    ggplot(aes(x = value)) +
    facet_wrap(~ key, scales = "free") +
    geom_histogram()

df %>%
    select_if(is.numeric) %>% 
    gather() %>% 
    ggplot(aes(x = value)) +
    facet_wrap(~ key, scales = "free") +
    stat_ecdf()
#----------------Numeric distribution-----------------------
num_vars <- df %>% select_if(is.numeric)

for (var in colnames(num_vars)){
    if (n_distinct(df[[var]]) <= 28){
        print(fitdistrplus::descdist(df[[var]], discrete = T, boot = 500, method = "sample"))
        print(title(var, outer = T, cex.main = 1, line = -3.5))
    }
    else{
        print(fitdistrplus::descdist(df[[var]], discrete = F, boot = 500, method = "sample"))
        print(title(var, outer = T, cex.main = 1, line = -3.5))
    }
}


#-------------------Bivariate analysis---------------------
#Categorical Vars

plot_var_Revenue<- function(x){
    group<- df %>% group_by(df[,x], Revenue) %>% summarize(n = n()) %>% 
        mutate(prop = n/sum(n)) %>% rename(var = "df[, x]")
    
    g0<-ggplot(group,aes(x = var, y = prop, fill = Revenue)) + 
        geom_col() +
        geom_text(aes(label = percent(prop)), vjust = -0.5) + 
        scale_y_continuous(labels = scales::percent) + 
        labs(x = x,
             title = paste("Revenue percent by",x)) + 
        theme(legend.position="bottom")
    
    prop_count<- df %>% group_by(df[,x]) %>% summarize(n = n()) %>% 
        mutate(prop = n/sum(n)) %>% rename(var = "df[, x]")
    
    g1<-ggplot(prop_count, aes(x = var, y=prop)) + 
        geom_col() + 
        geom_text(aes(label = percent(prop)), vjust = -0.5) +
            scale_y_continuous(labels = scales::percent) +
                labs(x = x,
                 title = paste("Visitors proportion across", x) ) +
        expand_limits(y = seq(0,1, by = .1))
        
    
    grid.arrange(g1, g0, nrow = 2, ncol =1)
}

plot_var_Revenue("Region")

plot_var_Revenue("Month")

plot_var_Revenue("Browser")

plot_var_Revenue("VisitorType")

plot_var_Revenue("SpecialDay")

plot_var_Revenue("OperatingSystems")

plot_var_Revenue("Weekend")
#-------------------------------------d
df %>% select_if(is.factor)
#Numerics vars
num_vars <- df %>% select_if(is.numeric)
#ggpairs(num_vars)
#-------------------
#The variables with the highest correlation

facet<-ggplot(df, aes(ProductRelated, ProductRelated_Duration, color = Revenue)) + 
     geom_point(alpha = .4) + geom_smooth(method = "lm") + facet_wrap(~ Revenue)

scatter<-ggplot(df, aes(ProductRelated, ProductRelated_Duration, color = Revenue)) + 
    geom_point(show.legend = F, alpha = .4) + geom_smooth(method = "lm", show.legend = F)

grid.arrange(facet, scatter, ncol = 1, nrow = 2)

cor.test(df$ProductRelated, df$ProductRelated_Duration)

#-------------------------------------------------------------

facet1<-ggplot(df, aes(BounceRates, ExitRates, color = Revenue)) + 
    geom_point(alpha = .4) + geom_smooth(method = "lm") + facet_wrap(~Revenue)

scatter1<-ggplot(df, aes(BounceRates, ExitRates, color = Revenue)) + 
    geom_point(show.legend = F, alpha = .4) + geom_smooth(method = "lm",show.legend = F)
grid.arrange(facet1, scatter1, ncol = 1, nrow = 2)

cor.test(df$BounceRates, df$ExitRates)

#--------------------------------------------------------
#------------Numeric variables distribution by Revenue-----------------

for (var in colnames(num_vars)){
    jit<- ggplot(df,aes(x = Revenue, y = df[,var])) + 
            geom_jitter(aes(color = Revenue),show.legend = F) + 
              labs(y = var,
                   title = "Jitterplot") +
                theme(axis.title.y = element_blank())
    
    box<- ggplot(df,aes(x = Revenue, y = df[,var])) + 
            geom_boxplot(aes(fill = Revenue),show.legend = F) + 
                labs(y = var,
                     title = "Boxplot")
    ecdf<- ggplot(df,aes(df[,var])) + 
            stat_ecdf(aes(color = Revenue)) +
                labs(x = var,
                     y = "Probs",
                     title = "Ecdf") + 
                expand_limits(x = seq(0, max(df[,var]), by = 1))
    
    print(grid.arrange(ecdf,arrangeGrob( box, jit ,ncol = 2), nrow = 2))
}

num_var_dist <- function(y){
    jit<- ggplot(df,aes(x = Revenue, y = df[[y]])) + 
        geom_jitter(aes(color = Revenue),show.legend = F) + 
        labs(y = y,
             title = "Jitterplot") +
        theme(axis.title.y = element_blank())
    
    box<- ggplot(df,aes(x = Revenue, y = df[[y]])) + 
        geom_boxplot(aes(fill = Revenue),show.legend = F) + 
        labs(y = y,
             title = "Boxplot")
    ecdf<- ggplot(df,aes(df[[y]])) + 
        stat_ecdf(aes(color = Revenue)) +
        labs(x = y,
             y = "Probs",
             title = "Ecdf") + 
        expand_limits(x = seq(0, max(df[[y]]), by = 1))
    
    print(grid.arrange(ecdf,arrangeGrob( box, jit ,ncol = 2), nrow = 2))
    
}

num_var_dist("ExitRates")




#--------------------------------------------------------------