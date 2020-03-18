if [ "$HOSTNAME" != "ttmagpie" ]; then
	echo "sí copio a /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/$1"
	cp $1 /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/$1
else
	echo 'no copio'
fi 
