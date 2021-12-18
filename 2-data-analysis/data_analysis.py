import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud

def plot_word_cloud(reports, severities, max_len=128):
    """Plots word clouds by severity in the severities list.

    Args:
        reports (dataframe): a bug reports dataframe.
        severities (list)  : a list of severity levels list.
        max_len (int)      : max number of words in each cloud.
        
    """
    def do_plot(ax, severity):
        filtered = reports.loc[reports['severity_code']==severity]
        descriptions = filtered['long_description'].apply(
            (lambda s: ' '.join(s.split()[:max_len]))
        )
        descriptions = " ".join(
            description for description in descriptions
        )
        wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=max_len
                            , background_color="white").generate(descriptions)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(severity)
        ax.grid(False)

    stopwords = set(STOPWORDS)
    if len(severities) > 1:
        fig, axs  = plt.subplots(nrows=1, ncols=len(severities), figsize=(20, 20))
        for ax, severity in zip(axs, severities):
            do_plot(ax, severity)
    else:
        fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
        do_plot(ax, severities[0])