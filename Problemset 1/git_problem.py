import sqlite3 as lite

def q1(con):
 # FIND Ra, Dec OF ALL OBJECTS WITH B > 16
 rows = con.execute('Select Name, Ra, Dec from MagTable WHERE B > 16')
 
 print 'Printing all objects with B > 16...'
 for row in rows:
  print '  {0}: Ra = {1} & Dec = {2}'.format(row[0], row[1], row[2])

def q2(con):
 # OUTPUT B, R, Teff FOR ALL STARS
 # NB Only select stars here that also occur in PhysTable
 rows = con.execute('Select m.Name, m.B, m.R, p.Teff, p.FeH from MagTable as m \
			JOIN PhysTable as p on m.Name=p.Name')

 print 'Printing stellar properties for all stars... that is, where Teff and FeH are defined...'
 for row in rows:
  print '   {0}: B = {1}, R = {2}, Teff = {3}, FeH = {4}'\
			.format(row[0],row[1],row[2],row[3],row[4])

def q3(con):
 # OUTPUT B, R, Teff, FeH FOR ALL STARS WITH FeH > 0
 # NB Only select stars here that also occur in PhysTable
 rows = con.execute('Select m.Name, m.B, m.R, p.Teff, p.FeH from MagTable as m \
			JOIN PhysTable as p on m.Name=p.Name where p.FeH>0')

 print 'Printing stellar properties for all stars with FeH > 0...'
 for row in rows:
  print '   {0}: B = {1}, R = {2}, Teff = {3}, FeH = {4}'\
			.format(row[0],row[1],row[2],row[3],row[4])

def q4(con):
 # CREATE TABLE WITH B-R COLOUR
 # PLACE IT IN CURRENT DIRECTORY
 rows = con.execute('Select Name, B-R as br FROM MagTable')

 br_table=open('BRTable.dat', 'w+')
 for row in rows:
  br_table.write('%s,%f\n'%(row[0],row[1]))

if __name__ == '__main__':
 con = lite.connect('Pr1.db')

 #q1(con)
 #q2(con)
 #q3(con)
 q4(con)
