"""The cleanup region is defined in the RGB image for both recognition and point enrollment."""


def overlaps_table(item, tables):
    left,top,right,bottom=item['box']
    return any(min(right,x1)>max(left,x0) and min(bottom,y1)>max(top,y0)
               for x0,y0,x1,y1 in (table['box'] for table in tables))
