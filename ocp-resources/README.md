# Install on OCP

## Create Secret containing the OpenAI token
```
oc create secret generic openaicreds --from-literal=password='sk-.....'
```

## Apply the resources to the OCP cluster
```
oc apply -f svcsbot-resources.yaml
```

## Get the SSL route and open with a browser
```
oc get route
```