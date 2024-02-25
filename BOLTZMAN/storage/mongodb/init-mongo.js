print("--------------------------");
print("Started Adding the Users.");
print("--------------------------");
printjson("Started Adding the Users.")
db = db.getSiblingDB("datastore");

db.createUser({
  user: "dataignite",
  pwd: "dataignite",
  roles: [{ role: "readWrite", db: "datastore" }],
});
db.getUsers()

print("--------------------------");
print("End Adding the User Roles.");
print("--------------------------");