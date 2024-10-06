image_name=eu.gcr.io/mehhala-fantasy/daily-fantasy
image_tag=latest
full_image_name=${image_name}:${image_tag}
docker build --rm -t "${full_image_name}" .
docker push "$full_image_name"
# Output the strict image name, which contains the sha256 image digest
docker inspect --format='{{.Config.Image}}' ${full_image_name}