library(ggbump) #install_github("davidsjoberg/ggbump")
library(ggplot2)
library(leaflet)
# データ準備
df <- data.frame(country=c(rep("A", 4), rep("B", 4), rep("C", 4), rep("D", 4)),
           year=rep(c(2017, 2018, 2019, 2020), 4),
           rank=c(4,2,1,3, 1,3,2,4, 3,1,3,2, 2,4,4,1))

ggplot(df, aes(year, rank, color = country)) +
  geom_point(size = 7) +
  geom_text(data = df %>% filter(year == min(year)),
            aes(x = year - .1, label = country), size = 5, hjust = 1) +
  geom_text(data = df %>% filter(year == max(year)),
            aes(x = year + .1, label = country), size = 5, hjust = 0) +
  geom_bump(size = 2, smooth = 8) +
  scale_x_continuous(limits = c(2016.6, 2020.4),
                     breaks = seq(2017, 2020, 1)) +
  theme(legend.position = "none",
        panel.grid.major = element_blank()) +
  labs(y = "RANK",
       x = NULL) +
  scale_y_reverse()